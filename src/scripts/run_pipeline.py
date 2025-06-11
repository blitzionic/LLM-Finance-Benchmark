import os
import sys
import glob
import argparse
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.pipeline import Pipeline
from RAG import get_query_engine

def run_pipeline(config: str, dataset_path: str, output_dir: str, model: str = "llama-2-70b-chat", **kwargs):
    """
    Run the pipeline on a dataset with the specified configuration.

    Args:
        config: Pipeline configuration (B-0 to B-3)
        dataset_path: Path to the dataset file
        output_dir: Directory to save results
        model: Model to use
        **kwargs: Additional provider-specific arguments
    """
    # load env variables 
    load_dotenv()
    
    # Ensure dataset file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # set up RAG if needed
    query_engine = None
    if config in ["B-3"]:
        query_engine = get_query_engine() 
    
    # Remove provider from kwargs since it's already in the config
    provider = kwargs.pop("provider")
    
    pipeline = Pipeline(
        config=config,
        provider=provider,
        model=model,
        query_engine=query_engine,
        **kwargs
    )
    
    # load current dataset 
    df = pd.read_csv(dataset_path)
    results = []
    
    # Process each question
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {config}"):
        # Format question with choices in a clear, structured format
        question_text = f"""
        {row['question']}
        A) {row['A']}
        B) {row['B']}
        C) {row['C']}
        D) {row['D']}
        """
        result = pipeline.process(question_text)
        # adds ground truth to the result
        result["ground_truth"] = row["answer"]
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    output_path = os.path.join(output_dir, f"{config}_results.csv")
    
    # Save results, overwriting if file exists
    results_df.columns = [f"{col}_{model}_{config}" for col in results_df.columns]
    results_df.to_csv(output_path, index=False)
    
    print(f"Results of model {model} and config {config} saved to {output_path}")

def get_provider_config(provider: str, model: str = None) -> dict:
    """
    Get the configuration for the specified provider.
    
    Args:
        provider: One of "openai", "runpod", "anthropic", "google"
        model: Optional model name to override the default
        
    Returns:
        dict: Provider configuration
    """
    load_dotenv()
    
    # Default models for each provider
    default_models = {
        "openai": "gpt-4o-mini",
        "runpod": "llama-2-70b-chat",
        "anthropic": "claude-3-opus-20240229",
        "google": "gemini-pro"
    }
    
    # Available models for each provider
    available_models = {
        "openai": ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "runpod": ["llama-2-70b-chat", "llama-2-13b-chat", "llama-2-7b-chat"],
        "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "google": ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-pro", "gemini-pro-vision"]
    }
    
    if provider not in available_models:
        raise ValueError(f"Unknown provider: {provider}")
        
    # Use provided model or default
    model = model or default_models[provider]
    if model not in available_models[provider]:
        raise ValueError(f"Model {model} not available for provider {provider}. Available models: {available_models[provider]}")
    
    config = {
        "provider": provider,
        "model": model
    }
    
    # Add provider-specific configurations
    if provider == "openai":
        config["api_key"] = os.getenv("OPENAI_API_KEY")
    elif provider == "runpod":
        config["endpoint"] = os.getenv("RUNPOD_ENDPOINT")
        config["api_key"] = os.getenv("RUNPOD_API_KEY")
    elif provider == "anthropic":
        config["api_key"] = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "google":
        config["api_key"] = os.getenv("GOOGLE_API_KEY")
        
    return config

def main():
    """Run the pipeline for all configurations and datasets"""
    # example: python run_pipeline.py --provider openai --model gpt-4o-mini --config B-0

    parser = argparse.ArgumentParser(description="Run the pipeline with different LLM providers")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "runpod", "anthropic", "google"],
        default="runpod",
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to use with the provider"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["B-0", "B-2", "B-3", "all"],
        default="all",
        help="Which configuration to run (B-0 to B-3, or 'all' for all configurations)"
    )
    args = parser.parse_args()
    
    # Get provider configuration
    provider_config = get_provider_config(args.provider, args.model)
    
    # Ask user about dataset selection
    print("\nDataset Selection:")
    print("1. Run on a specific file")
    print("2. Run on all CSV files in question_sheets")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    # Get path relative to root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datasheets_directory = os.path.join(root_dir, "data", "question_sheets")
    
    if choice == "1":
        csv_files = glob.glob(os.path.join(datasheets_directory, "*.csv"))
        print("\nAvailable datasets:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {os.path.basename(file)}")
        
        file_choice = input("\nEnter the number of the file you want to use: ").strip()
        try:
            file_idx = int(file_choice) - 1
            if 0 <= file_idx < len(csv_files):
                dataset_files = [csv_files[file_idx]]
            else:
                print("Invalid file number. Exiting.")
                return
        except ValueError:
            print("Invalid input. Exiting.")
            return
            
    elif choice == "2":
        # Get all CSV files
        dataset_files = glob.glob(os.path.join(datasheets_directory, "*.csv"))
        if not dataset_files:
            print("No CSV files found in data/question_sheets/")
            return
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Determine which configs to run
    if args.config == "all":
        configs = ["B-0", "B-2", "B-3"]
    else:
        configs = [args.config]
    
    for dataset_file in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        output_dir = os.path.join(root_dir, "data", "results", dataset_name, args.provider, provider_config['model'])
        
        print(f"\nProcessing dataset: {dataset_name}")
        print(f"Using provider: {args.provider}")
        print(f"Using model: {provider_config['model']}")
        
        for config in configs:
            print(f"\nRunning {config} with {args.provider}")
            try:
                run_pipeline(
                    config=config,
                    dataset_path=dataset_file,
                    output_dir=output_dir,
                    **provider_config
                )
            except Exception as e:
                print(f"Error running {config}: {e}")

if __name__ == "__main__":
    main() 