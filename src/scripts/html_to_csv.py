import os
import csv
from bs4 import BeautifulSoup
import re

input_folder = "./data/temp_html"
output_folder = "./data/question_sheets"
os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
output_file = os.path.join(output_folder, "dividend .csv")
 

header = ["id","question","A","B","C","D","Answer","Explanation","Hint"]

existing_questions = set()
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            existing_questions.add(row["question"])

def parse_html_file(filepath, question_id):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            html_content = f.read()
    except Exception as e:
        print(f"ERROR reading file {filepath}: {e}")
        return None

    try:
        soup = BeautifulSoup(html_content, "html.parser")
    except Exception as e:
        print(f"CAUTION: Error parsing HTML for file {filepath}: {e}")
        return None

    question_h2 = soup.find("h2", string=lambda s: s and "Question:" in s)
    if not question_h2:
        print(f"CAUTION: No 'Question:' heading found in file {filepath}. Skipping.")
        return None

    question_div = question_h2.find_next("div")
    if not question_div:
        print(f"CAUTION: No question <div> found after 'Question:' heading in file {filepath}. Skipping.")
        return None

    p_tags = question_div.find_all("p")
    if not p_tags:
        print(f"CAUTION: No <p> tags found under question section in file {filepath}. Skipping.")
        return None

    question_text = ""

    # Handle "True or False" type questions specifically
    first_tag_text = p_tags[0].get_text(strip=True)
    if "true or false" in first_tag_text.lower():
        # Append "True or False:" and the subsequent statement
        question_text = first_tag_text
        if len(p_tags) > 1:  # Ensure there's a second paragraph for the statement
            question_text += " " + p_tags[1].get_text(strip=True)
    else:
        # For other types of questions, use the first paragraph as the question text
        question_text = first_tag_text

    # check if question is duplicate (word for word match)
    if question_text in existing_questions:
        print(f"DUPLICATE question found in file {filepath}. Skipping.")
        return None
    
    is_true_false = False

    answers = {}
    # Extract possible multiple-choice answers
    for p in p_tags[1:]:
        line = p.get_text(strip=True)
        if line.lower().startswith(("a.", "a)")):     
            answers["A"] = line[2:].strip()
        elif line.lower().startswith(("b.", "b)")):
            answers["B"] = line[2:].strip()
        elif line.lower().startswith(("c.", "c)")):
            answers["C"] = line[2:].strip()
        elif line.lower().startswith(("d.", "d)")):
            answers["D"] = line[2:].strip()
        elif line.lower().startswith(("e.", "e)")):
            answers["E"] = line[2:].strip()

    if "true or false" in question_text.lower():
        is_true_false = True
    elif "A" in answers and "B" in answers:
        if "true" in answers["A"].lower() and "false" in answers["B"].lower():
            is_true_false = True

    if is_true_false:
        # For True/False questions
        answers = {"A": "True", "B": "False"}

    if "E" in answers:
        print(f"OPTION E: Question in file {filepath} has five options (A to E).")
        return None

    #print(f"true or false question type: {is_true_false}")
    # Check if it's free response
    
    if not is_true_false and len(answers) == 0:
        # print(f"is_true_false: {is_true_false}, length: {len(answers)}")
        # It's free response, do not include this question at all
        print(f"FRQ question type in file {filepath}. Skipping")
        return None

    # Find the "Answer and Explanation" heading by searching all h2 tags
    ans_exp_h2 = None
    for h2 in soup.find_all("h2"):
        if "Answer and Explanation" in h2.get_text():
            ans_exp_h2 = h2
            break

    # Find the answer_content div
    answer_div = soup.find("div", {"test-id":"answer_content"})
    if not answer_div:
        print(f"CAUTION: 'answer_content' div found in file {filepath}. Skipping.")
        return None

    # Extract hint
    hint_text = ""
    if ans_exp_h2:
        # Extract hint between question_div and ans_exp_h2
        curr_elem = question_div.find_next_sibling()
        while curr_elem and curr_elem != ans_exp_h2:
            if curr_elem.name in ["h2", "p"]:
                text = curr_elem.get_text(strip=True)
                if text:
                    hint_text += (text + " ")
            curr_elem = curr_elem.find_next_sibling()
    else:
        # No ans_exp_h2 found, extract hint between question_div and answer_div
        curr_elem = question_div.find_next_sibling()
        while curr_elem and curr_elem != answer_div:
            if curr_elem.name in ["h2", "p"]:
                text = curr_elem.get_text(strip=True)
                if text:
                    hint_text += (text + " ")
            curr_elem = curr_elem.find_next_sibling()
    hint_text = hint_text.strip()

    # Extract explanation
    explanation_text = answer_div.get_text("\n", strip=True)
    explanation_lines = explanation_text.split('\n')
    
    correct_letter = ""
    # print(explanation_lines)
    # If multiple choice or True/False, try to find correct answer
    if is_true_false:
        for p in answer_div.find_all("p"):
            text = p.get_text(strip=True).lower()
            if "true" in text and "false" not in text:
                correct_letter = "A"
                break
            elif "false" in text and "true" not in text:
                correct_letter = "B"
                break
      
    else: 
        correct_letter = None  # Initialize to None before the loop
        
        for i, line in enumerate(explanation_lines):
            lower_line = line.lower().strip()
        # if "answer" in lower_line or "option" in lower_line:  # Likely multiple choice
            # Check if the answer is directly in the lower_line
            # print(f"is b. in??? {'b.' in lower_line}")
            for choice in ['a.', 'b.', 'c.', 'd.', 'a:', 'b:', 'c:', 'd:', '(a)', '(b)', '(c)', '(d)', 'a)', 'b)', 'c)', 'd)']:
                if choice in lower_line:
                    # print("HEREEE")
                    correct_letter = choice[1].upper() if len(choice) > 2 else choice[0].upper()
                    break  # Found the correct letter, exit inner loop

            # If correct_letter found, exit the outer loop
            if correct_letter:
                break
            
            # Otherwise, check the next line for the answer
            if i + 1 < len(explanation_lines):
                next_line = explanation_lines[i + 1].strip().lower()
                for choice in ['a.', 'b.', 'c.', 'd.', 'a:', 'b:', 'c:', 'd:', '(a)', '(b)', '(c)', '(d)', 'a)', 'b)', 'c)', 'd)']:
                    if choice in next_line:
                        correct_letter = choice[1].upper() if len(choice) > 2 else choice[0].upper()
                        break  # Found the correct letter, exit inner loop

            # If correct_letter is found after checking the next line, exit the outer loop
            if correct_letter:
                break
            else:
                # Continue to the next iteration for non-answer lines
                continue

    # For True/False if no letter found
    if is_true_false and not correct_letter:
        lower_exp = explanation_text.lower()
        if "the correct answer is true" in lower_exp or "the correct option is true" in lower_exp:
            correct_letter = "A"
        elif "the correct answer is false" in lower_exp or "the correct option is false" in lower_exp:
            correct_letter = "B"
    
    # print(f"correct letter: {correct_letter}")
    # If still no correct_letter and it's MC, print a warning but continue
    if not correct_letter and len(answers) > 0:
        print(f"correct_letter: {correct_letter} and length: {len(answers)}")
        print(f"ERROR: could not determine correct answer letter in file {filepath}, skipping.")
        return None 

    A = answers.get("A","")
    B = answers.get("B","")
    C = answers.get("C","")
    D = answers.get("D","")

    return [question_id, question_text, A, B, C, D, correct_letter, explanation_text, hint_text]

try:
    file_exists = os.path.exists(output_file)

    # Open in append mode so we can run multiple times without overwriting
    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)

        with open(output_file, "r", encoding="utf-8") as f:
            row_count = sum(1 for line in f)  # Exclude the header
        current_id = row_count + 1  # Start ID from the next available row number

        for filename in os.listdir(input_folder):
            if filename.endswith(".html"):
                filepath = os.path.join(input_folder, filename)
                row = parse_html_file(filepath, current_id)
                if row:
                    try:
                        writer.writerow(row)
                        print(f"Successfully wrote question '{row[1]}' to CSV.")
                        current_id += 1
                    except Exception as e:
                        print(f"ERROR writing to CSV for file {filename}: {e}")
    
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        complete_questions = sum(
            1 for row in reader 
            if row["question"] and row["Answer"] and row["question"].strip() and row["Answer"].strip()
        )
    print(f"Exiting with {complete_questions} complete questions in the CSV file.")
except Exception as e:
    print(f"Error opening or writing to CSV file {output_file}: {e}")
