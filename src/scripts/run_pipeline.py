import os
import openai 
import sys
import glob
import json
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from llama_index.core import StorageContext, load_index_from_storage
from agents.pipeline import Pipeline
from utils import calculate_score

def load_topic_roles():
    """Load the topic roles from the JSON file"""
    roles_path = os.path.join(project_root, "config/topic_roles.json")
    with open(roles_path, 'r') as f:
        return json.load(f)

def setup_rag():
    # load and return the index from the storage
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
    return index.as_query_engine()

# run through one dataset based on one [B-0, B-1, B-2, B-3] configuration
def run_pipeline(config: str, dataset_path: str, output_dir: str, model: str, provider: str = "openai", **kwargs):
    """
    Run the pipeline on a dataset with the specified configuration.
    
    Args:
        config: Pipeline configuration (B-0 to B-3)
        dataset_path: Path to the dataset file
        output_dir: Directory to save results
        model: Model name/identifier
        provider: LLM provider to use ("openai", "runpod")
        **kwargs: Additional provider-specific arguments
    """
    
    # load variables
    load_dotenv() 
    
    # Set up RAG if needed
    query_engine = None
    if config in ["B-1", "B-3"]:
        print("Setting up RAG")
        query_engine = setup_rag()
    
    # Load role description only for configurations that use the critic reviewer
    role_description = None
    if config in ["B-2", "B-3"]:
        topic_roles = load_topic_roles()
        topic_name = os.path.splitext(os.path.basename(dataset_path))[0].lower()
        role_description = topic_roles.get(topic_name)
        if not role_description: # throw error if no role description found
            raise ValueError(f"No role description found for topic '{topic_name}'")
    
    # create a pipeline for the current model and config
    pipeline = Pipeline(
        config=config,
        provider=provider,
        model=model,
        query_engine=query_engine,
        role_description=role_description,
        **kwargs
    )
    
    # load current question csv file 
    df = pd.read_csv(dataset_path)
    results = []
    
    # Process each question
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {config}"):
        # Format question with multiple choice options
        question = (
            f"{row['question']}\n"
            f"A: {row['A']}\n"
            f"B: {row['B']}\n"
            f"C: {row['C']}\n"
            f"D: {row['D']}\n"
        )
        print(f"Processing question: {question}")
        result = pipeline.process(question)
        result["ground_truth"] = row["answer"]
        results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy
    # accuracy = calculate_score(results_df)
    # print(f"\nConfiguration {config} accuracy: {accuracy:.2%}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{config}_{provider}_{model}_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    

def main():
    """Run the pipeline for all configurations and datasets"""
    # load dataset file - use path relative to project root
    dataset_file = os.path.join(project_root, "data/question_sheets/Asset.csv")  
    dataset_files = [dataset_file]
    
    # Run each configuration
    configs = ["B-0", "B-1", "B-2", "B-3"]
    
    # OpenAI configuration
    openai_config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY")
    }

    # RunPod configuration
    runpod_config = {
        "provider": "runpod",
        "model": "llama-2-70b-chat",
        "endpoint": os.getenv("RUNPOD_ENDPOINT"),
        "api_key": os.getenv("RUNPOD_API_KEY")
    }
    
    # Choose which provider to use
    provider_choice = input("Choose provider (openai/runpod): ").strip().lower()
    if provider_choice == "openai":
        config_to_use = openai_config
    else:
        config_to_use = runpod_config
    
    for dataset_file in dataset_files:
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        output_dir = os.path.join(project_root, f"results/{dataset_name}")
        
        print(f"\nProcessing dataset: {dataset_name}")
        
        for config in configs:
            print(f"\nRunning {config} with {config_to_use['provider']} {config_to_use['model']}")
            try:
                # Extract provider and model, pass the rest as kwargs
                provider = config_to_use.pop("provider")
                model = config_to_use.pop("model")
                run_pipeline(config=config, dataset_path=dataset_file, output_dir=output_dir, model=model, provider=provider, **config_to_use)
            except Exception as e:
                print(f"Error running {config}: {e}")
            finally:
                # Restore the config for next iteration
                config_to_use["provider"] = provider
                config_to_use["model"] = model

if __name__ == "__main__":
    main() 