import pandas as pd
import json

def load_dataset(file_path):
    """
    Loads the dataset from a CSV file.
    Parameters:
        file_path (str): Path to the CSV file containing the dataset.
    Returns:
        DataFrame: The loaded dataset.
    """
    try:
        dataset = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}.")
        return dataset
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def save_results(results, output_path):
    """
    Saves the results DataFrame to a CSV file.
    Parameters:
        results (DataFrame): The DataFrame to save.
        output_path (str): The file path for the output CSV.
    """
    try:
        results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}.")
    except Exception as e:
        print(f"Error saving results to {output_path}: {e}")
        raise

def calculate_score(results, answer_column="answer", guess_column="guess"):
    try:
        correct = (results[guess_column] == results[answer_column]).sum()
        total = len(results)
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Accuracy calculated: {accuracy:.2f}%")
        return accuracy
    except KeyError as e:
        print(f"Error: Missing column in dataset: {e}")
        raise
    except Exception as e:
        print(f"Error calculating score: {e}")
        raise

def load_config(file_path="config.json"):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file {file_path} not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file: {e}")
        exit(1)
        
def get_profile(config, profile_name=None):
    """Retrieve the selected profile or the default profile."""
    profiles = config.get("profiles", {})
    if profile_name:
        profile = profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found in config.")
        return profile
    # Fallback to default profile
    default_profile = config.get("default_profile")
    if default_profile:
        return profiles.get(default_profile)
    raise ValueError("No default profile specified in the configuration.")

