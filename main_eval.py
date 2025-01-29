import argparse
import os
import pandas as pd
from tqdm import tqdm
from utils.utils import load_config, load_dataset, save_results, calculate_score, get_profile
from cot_eval import ask_gpt_with_cot
from gpt_eval import ask_gpt_direct
from openai import OpenAI


def benchmark_direct(dataset, client):
    """Benchmarks the dataset using direct-answer querying."""
    pbar = tqdm(dataset.iterrows(), total=len(dataset), desc="Benchmarking Direct Answers")
    for index, row in pbar:
        question = f"""Question:
{row['question']}
A. {row['A']}
B. {row['B']}
C. {row['C']}
D. {row['D']}
Please choose the correct answer: A, B, C, or D."""
        
        # Query the direct method with structured output
        answer = ask_gpt_direct(question, client, model="gpt-4o")
        dataset.loc[index, "baseline_guess"] = answer.strip()  # Save only the letter
        
        # Update progress bar with current score
        correct = (dataset["baseline_guess"] == dataset["answer"]).sum()
        pbar.set_postfix({"Correct": f"{correct}/{index + 1}"})

        # Print the model's answer and the correct answer
        print(f"Question {index + 1}: {row['question']}")
        print(f"Model's Answer: {answer.strip()}")
        print(f"Correct Answer: {row['answer']}\n{'-'*40}")
    return dataset


def benchmark_with_cot(dataset, client):
    """Benchmarks the dataset using Chain of Thought reasoning."""
    pbar = tqdm(dataset.iterrows(), total=len(dataset), desc="Benchmarking CoT Reasoning")
    for index, row in pbar:
        question = f"""Question:
{row['question']}
A. {row['A']}
B. {row['B']}
C. {row['C']}
D. {row['D']}
Please choose the correct answer: A, B, C, or D."""
        
        # Query with CoT
        steps, final_answer = ask_gpt_with_cot(question, client)
        if final_answer:
            dataset.loc[index, "guess"] = final_answer
            dataset.loc[index, "reasoning"] = "\n".join(
                [step.explanation for step in steps]
            )
        else:
            dataset.loc[index, "guess"] = "N"  # Default for failed responses
            dataset.loc[index, "reasoning"] = "No response"

        # Update progress bar with current score
        correct = (dataset["guess"] == dataset["answer"]).sum()
        pbar.set_postfix({"Correct": f"{correct}/{index + 1}"})
    return dataset


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="Input CSV file with questions")
    parser.add_argument("-m", "--method", choices=["direct", "cot"], default="direct", help="Answering method")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    dataset = load_dataset(args.file)
    dataset["guess"] = ""
    dataset["reasoning"] = ""  # CoT requires reasoning steps

    config_file = "config.json"
    config = load_config(config_file)

    selected_profile_name = input("Enter profile name (or leave blank for default): ").strip()
    profile = get_profile(config, selected_profile_name)
    if not profile:
        print("No valid profile found. Exiting.")
        exit(1)

    api_key = profile.get("openai_api_key")
    if not api_key:
        print("Error: OpenAI API key not found in the selected profile.")
        exit(1)
    client = OpenAI(api_key=api_key)

    if args.method == "cot":
        print("Using Chain of Thought reasoning...")
        results = benchmark_with_cot(dataset, client)
    else:
        print("Using direct-answer method...")
        results = benchmark_direct(dataset, client)

    accuracy = calculate_score(results)
    print(f"Final Accuracy: {accuracy:.2f}%")

    save_results(results, args.output)


if __name__ == "__main__":
    main()
