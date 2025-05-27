import os
import openai 
import sys
import glob
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from llama_index import StorageContext, load_index_from_storage
from agents.pipeline import Pipeline
from utils import calculate_score

def setup_rag():
    """Set up the RAG query engine"""
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
    return index.as_query_engine()

def run_pipeline(
    config: str,
    dataset_path: str,
    output_dir: str,
    model: str = "llama-2-70b-chat",
    **kwargs
):
    """
    Run the pipeline on a dataset with the specified configuration.
    
    Args:
        config: Pipeline configuration (B-0 to B-3)
        dataset_path: Path to the dataset file
        output_dir: Directory to save results
        model: Llama model to use
        **kwargs: Additional provider-specific arguments
    """
    # Load environment variables
    load_dotenv()
    
    # Set up RAG if needed
    query_engine = None
    if config in ["B-1", "B-3"]:
        query_engine = setup_rag()
    
    
    pipeline = Pipeline(
        config=config,
        provider="runpod",
        model=model,
        query_engine=query_engine,
        **kwargs
    )
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    results = []
    
    # Process each question
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {config}"):
        result = pipeline.process(row["question"])
        result["ground_truth"] = row["answer"]
        results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy
    accuracy = calculate_score(results_df)
    print(f"\nConfiguration {config} accuracy: {accuracy:.2%}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{config}_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main():
    """Run the pipeline for all configurations and datasets"""
    # Get dataset file
    dataset_file = "data/question_sheets/Asset.csv" 
    dataset_files = [dataset_file]
    
    # Run each configuration
    configs = ["B-0", "B-1", "B-2", "B-3"]
    
    # RunPod configuration
    runpod_config = {
        "model": "llama-2-70b-chat",
        "endpoint": os.getenv("RUNPOD_ENDPOINT"),
        "api_key": os.getenv("RUNPOD_API_KEY")
    }
    
    for dataset_file in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        output_dir = f"results/{dataset_name}"
        
        print(f"\nProcessing dataset: {dataset_name}")
        
        for config in configs:
            print(f"\nRunning {config} with RunPod Llama")
            try:
                run_pipeline(
                    config=config,
                    dataset_path=dataset_file,
                    output_dir=output_dir,
                    **runpod_config
                )
            except Exception as e:
                print(f"Error running {config}: {e}")

if __name__ == "__main__":
    main() 