import csv

def read_questions_from_csv(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            
            if reader.fieldnames is None:
                print("ERROR: No headers found in the CSV file. Check the file format.")
                return

            print(f"Reading questions from: {file_path}")
            print("-" * 80)
            
            for idx, row in enumerate(reader, start=1):
                print(f"Question {idx}:")
                print(f"  ID: {row.get('id', '').strip()}")
                print(f"  Question: {row.get('question', '').strip()}")
                print(f"  Choice A: {row.get('A', '').strip()}")
                print(f"  Choice B: {row.get('B', '').strip()}")
                print(f"  Choice C: {row.get('C', '').strip()}")
                print(f"  Choice D: {row.get('D', '').strip()}")
                print(f"  Correct Answer: {row.get('Answer', '').strip()}")
                print(f"  Explanation: {row.get('Explanation', '').strip()}")
                print(f"  Hint: {row.get('Hint', '').strip()}")
                print("-" * 80)

    except FileNotFoundError:
        print(f"ERROR: File not found at path '{file_path}'. Please check the file location.")
    except Exception as e:
        print(f"ERROR reading the CSV file: {e}")

# Specify the path to your CSV file
csv_file_path = "./data/question_sheets/Balance Sheets.csv"

# Call the function to read and display questions
read_questions_from_csv(csv_file_path)
