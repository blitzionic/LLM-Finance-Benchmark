import os
import pickle
import re
import time
import faiss
import json
import requests
import openai
import tiktoken
import pdfplumber
import xml.etree.ElementTree as ET
import speech_recognition as sr
import numpy as np
from bs4 import BeautifulSoup
from docx import Document
from pydub import AudioSegment
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from mistralai import Mistral
from together import Together

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

openai.api_key = config['openai_api_key']
MISTRAL_API_KEY = config['mistral_api_key']
TOGETHER_API_KEY = config['together_ai_api_key']

# Ollama REST API endpoint (if using Ollama)
OLLAMA_API_URL = 'http://localhost:11434/api/generate'

# Initialize the Mistral client
client = Mistral(api_key=MISTRAL_API_KEY)

together_client = Together(api_key=TOGETHER_API_KEY) 

GOOGLE_DOC_ID = config['google_doc_id']

def fetch_question_from_google_doc(doc_id):
    """
    Fetches the text content from a Google Doc by its document ID.
    """
    creds = Credentials.from_authorized_user_file('credentials.json')
    service = build('docs', 'v1', credentials=creds)
    
    document = service.documents().get(documentId=doc_id).execute()
    doc_content = document.get('body').get('content')

    question_text = ''
    for element in doc_content:
        if 'paragraph' in element:
            for text_element in element['paragraph']['elements']:
                if 'textRun' in text_element:
                    question_text += text_element['textRun']['content']

    return question_text.strip()

def parse_pdfs(pdf_file):
    """
    Parses a PDF file and extracts its text content.
    """
    all_text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + '\n'
    return all_text

def parse_xml(xml_file):
    """
    Parses an XML file and extracts its text content.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return ET.tostring(root, encoding='utf8', method='text').decode('utf-8')

def parse_json(json_file):
    """
    Parses a JSON file and returns its content as a formatted string.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return json.dumps(data, indent=4)

def parse_html(html_file):
    """
    Parses an HTML file and extracts its text content.
    """
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    return soup.get_text()

def parse_docx(docx_file):
    """
    Parses a DOCX file and extracts its text content.
    """
    doc = Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def parse_txt(txt_file):
    """
    Parses a TXT file and extracts its text content.
    """
    with open(txt_file, 'r', encoding='utf-8') as f:
        return f.read()

def parse_audio(audio_file):
    """
    Parses an audio file and extracts its text content using OpenAI's Whisper API.
    Supported formats: .mp3, .wav, .m4a, etc.
    """
    print(f"Transcribing audio file: {audio_file}")
    recognizer = sr.Recognizer()
    
    # Convert audio to a format compatible with the speech_recognition library if necessary
    file_extension = os.path.splitext(audio_file)[1].lower()
    if file_extension not in ['.wav', '.flac', '.aiff']:
        # Convert to WAV using pydub
        audio = AudioSegment.from_file(audio_file)
        converted_audio_file = os.path.splitext(audio_file)[0] + '.wav'
        audio.export(converted_audio_file, format='wav')
        audio_file = converted_audio_file
        print(f"Converted audio to WAV format: {converted_audio_file}")
    
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            # Use OpenAI's Whisper API for transcription
            transcription = openai.Audio.transcribe(
                model="whisper-1",
                file=open(audio_file, "rb")
            )
            return transcription['text']
        except Exception as e:
            print(f"Error transcribing audio file {audio_file}: {e}")
            return ""

def parse_files(file_dir, output_file):
    print("Began parsing files.")
    all_text = ''

    total_files = sum([len(files) for r, d, files in os.walk(file_dir)])
    print(f"Total files to parse: {total_files}")

    counter = 0

    # Walk through all subdirectories and files
    for root, _, files in os.walk(file_dir):
        for file in files:
            counter += 1
            print(f"Parsed file {counter}/{total_files}: {file}")

            file_path = os.path.join(root, file)
            if file.endswith('.pdf'):
                all_text += parse_pdfs(file_path) + '\n'
            elif file.endswith('.xml'):
                all_text += parse_xml(file_path) + '\n'
            elif file.endswith('.json'):
                all_text += parse_json(file_path) + '\n'
            elif file.endswith('.html') or file.endswith('.htm'):
                all_text += parse_html(file_path) + '\n'
            elif file.endswith('.docx'):
                all_text += parse_docx(file_path) + '\n'
            elif file.endswith('.txt'):
                all_text += parse_txt(file_path) + '\n'
            elif file.endswith(('.mp3', '.wav', '.m4a', '.flac', '.aiff')):
                transcription = parse_audio(file_path)
                if transcription:
                    all_text += transcription + '\n'
            else:
                print(f"Unsupported file type: {file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(all_text)

    print("Finished parsing files.")

def preprocess_text(input_file):
    """
    Reads text from a file and performs preprocessing steps.
    """
    print("Began preprocessing text.")
    with open(input_file, 'r', encoding='utf-8') as f:
        text_content = f.read()

    # Remove multiple newlines and replace them with spaces
    text_content = re.sub(r'\n+', ' ', text_content)

    # Remove headers and footers (adjust regex patterns as needed)
    text_content = re.sub(r'Page \d+ of \d+', '', text_content)

    # Remove extra spaces and tabs
    text_content = re.sub(r'\s+', ' ', text_content).strip()

    print("Finished preprocessing text.")
    return text_content

def count_tokens(text, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_text(text, max_length=4000, overlap=1500):
    """
    Splits text into chunks based on token count with overlap to maintain context.
    """
    print("Began chunking text.")
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = encoding.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_length - overlap
    
    print("Finished chunking text.")
    return chunks

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generates an embedding for the given text using OpenAI's API.
    """
    response = openai.Embedding.create(
        input=[text],
        model=model
    )
    return response['data'][0]['embedding']

def generate_embeddings(chunks):
    embeddings = []
    total_chunks = len(chunks)
    processing_rate = 3 # chunks per second; varies by usage tier (https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-five)
    cost_per_token = 0.1/1000000 # $0.1 per million tokens/1 million tokens

    print("Started embedding generation.")
    start_time = time.time()
    total_cost = 0

    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings.append(embedding)

        remaining_chunks = total_chunks - (idx + 1)
        estimated_remaining_time = remaining_chunks / processing_rate  # in seconds

        hours = int(estimated_remaining_time // 3600)
        minutes = int((estimated_remaining_time % 3600) // 60)
        seconds = int(estimated_remaining_time % 60)

        tokens = count_tokens(chunk)
        cost = tokens * cost_per_token
        total_cost += cost

        print(f'Processing chunk {idx + 1}/{total_chunks}. Estimated time remaining: {hours}h {minutes}m {seconds}s. Running total cost: ${total_cost:.5f}.')

    total_elapsed = time.time() - start_time
    total_hours = int(total_elapsed // 3600)
    total_minutes = int((total_elapsed % 3600) // 60)
    total_seconds = int(total_elapsed % 60)
    print(f"Finished embedding generation in {total_hours}h {total_minutes}m {total_seconds}s.")
    return embeddings

def create_faiss_index(embeddings):
    """
    Creates a FAISS index from embedding vectors.
    """
    embedding_dim = len(embeddings[0])
    embedding_matrix = np.array(embeddings).astype('float32')

    # Create a FAISS index
    index = faiss.IndexFlatL2(embedding_dim)

    # Add embeddings to the index
    index.add(embedding_matrix)

    print(f'Number of vectors in the index: {index.ntotal}')
    return index

def save_index_and_chunks(index, chunks, index_file='faiss_index.idx', chunks_file='chunks.pkl'):
    """
    Saves the FAISS index and text chunks to disk.
    """
    faiss.write_index(index, index_file)
    with open(chunks_file, 'wb') as f:
        pickle.dump(chunks, f)

def load_index_and_chunks(index_file='faiss_index.idx', chunks_file='chunks.pkl'):
    """
    Loads the FAISS index and text chunks from disk.
    """
    index = faiss.read_index(index_file)
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

def embed_query(query, model="text-embedding-ada-002"):
    """
    Generates an embedding for the query text.
    """
    response = openai.Embedding.create(
        input=[query],
        model=model
    )
    return response['data'][0]['embedding']

def retrieve_chunks(query, index, chunks, k=5):
    """
    Retrieves the most relevant text chunks for a given query.
    """
    query_embedding = embed_query(query)
    query_vector = np.array([query_embedding]).astype('float32')

    # Search the index
    _, indices = index.search(query_vector, k)
    results = [chunks[i] for i in indices[0]]

    return results

def generate_answer(query, relevant_chunks):
    """
    Generates an answer to the query using GPT-4o and the relevant context.
    """
    context = '\n\n'.join(relevant_chunks)
    prompt = f"""You are a Corporate Finance expert. Answer the questions. For multiple choice, do not provide explanations only answer with answer choice description.

    Context:
    {context}

    {query}

    Answer as thoroughly as possible based on the context provided."""

        # If using Ollama API (uncomment the following code and comment out the above code)

    # payload = {
    #     "model": "llama3.1",
    #     "prompt": prompt
    # }

    # try:
    #     # Send the request to the Ollama API with stream=True
    #     response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
    #     response.raise_for_status()

    #     # Initialize the answer
    #     answer = ''

    #     # Read the response stream line by line
    #     for line in response.iter_lines():
    #         if line:
    #             try:
    #                 # Decode the line (if necessary)
    #                 decoded_line = line.decode('utf-8')
    #                 # Parse the JSON object
    #                 data = json.loads(decoded_line)
    #                 # Append the response text
    #                 answer += data.get('response', '')
    #             except json.JSONDecodeError as e:
    #                 print(f"JSON decode error: {e}")
    #                 continue

    #     return answer.strip()

    # except requests.exceptions.RequestException as e:
    #     print(f"Error: {e}")
    #     return "An error occurred while generating the answer."

    # If using OpenAI's GPT-4o
    # response = openai.ChatCompletion.create(
    #     model='gpt-4o',
    #     messages=[
    #         {'role': 'user', 'content': prompt}
    #     ],
    #     temperature=0.2,  # Lower temperature for more precise answers
    #     max_tokens=500
    # )
    
    # answer = response['choices'][0]['message']['content']
    # return answer.strip()

    # Mistral Large 2
    try:
        # Generate response using Mistral
        chat_response = client.chat.complete(
            model='mistral-large-latest',
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )

        # Extract the response content
        answer = chat_response.choices[0].message.content
        return answer.strip()

    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while generating the answer."
    
    # try:
    #     # Use Together AI's chat completion API
    #     response = together_client.chat.completions.create(
    #         model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",  # Specify the Llama model
    #         messages=[
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0.2,  # Adjust as needed
    #         max_tokens=2500,    # Adjust as needed
    #         stream=False        # Set to True if you want streaming responses
    #     )
    #     answer = response.choices[0].message.content.strip()
    #     return answer

    # except Exception as e:
    #     print(f"Error generating answer with Together AI: {e}")
    #     return "An error occurred while generating the answer."
    
def normal_chat_mode(query):
    """
    Generates an answer to the query directly without using retrieved chunks.
    """
    prompt = f"""You are a Financial Engineering expert providing detailed and accurate answers. For multiple choice questions, only provide the answer choice and answer. No explanations.

    {query}

    Answer as thoroughly as possible."""

        # If using Ollama API (uncomment the following code and comment out the above code)
    # payload = {
    #     "model": "llama3.1",
    #     "prompt": prompt
    # }

    # try:
    #     response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
    #     response.raise_for_status()

    #     answer = ''
    #     for line in response.iter_lines():
    #         if line:
    #             try:
    #                 decoded_line = line.decode('utf-8')
    #                 data = json.loads(decoded_line)
    #                 answer += data.get('response', '')
    #             except json.JSONDecodeError as e:
    #                 print(f"JSON decode error: {e}")
    #                 continue

    #     return answer.strip()

    # except requests.exceptions.RequestException as e:
    #     print(f"Error: {e}")
    #     return "An error occurred while generating the answer."

    # OpenAI's GPT-4o
    response = openai.ChatCompletion.create(
        model='gpt-4o',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.2,  # Lower temperature for more precise answers
        max_tokens=500
    )
    
    # answer = response['choices'][0]['message']['content']
    # return answer.strip()

    # Mistral Large 2
    # try:
    #     # Generate response using Mistral
    #     chat_response = client.chat.complete(
    #         model='mistral-large-latest',
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": prompt,
    #             },
    #         ]
    #     )

    #     # Extract the response content
    #     answer = chat_response.choices[0].message.content
    #     return answer.strip()

    # except Exception as e:
    #     print(f"Error: {e}")
    #     return "An error occurred while generating the answer."
    
    # try:
    #     # Use Together AI's chat completion API
    #     response = together_client.chat.completions.create(
    #         model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",  # Specify the Llama model
    #         messages=[
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0.2,  # Adjust as needed
    #         max_tokens=2500,    # Adjust as needed
    #         stream=False        # Set to True if you want streaming responses
    #     )
    #     answer = response.choices[0].message.content.strip()
    #     return answer

    # except Exception as e:
    #     print(f"Error generating answer with Together AI: {e}")
    #     return "An error occurred while generating the answer."


def main():
    index, chunks = None, None
    run_mode = input("Do you want to run RAG or normal mode? (rag/normal): ").strip().lower()

    if run_mode == 'rag':
        update_rag = input("Do you want to update the RAG components? (y/n): ").strip().lower()

        if update_rag == 'y' or update_rag == '':
            # Paths and filenames
            file_dir = 'Finance/International_Finance'
            text_output_file = 'international_finance.txt'

            # Step 1: Parse documents and extract text
            parse_files(file_dir, text_output_file)

            # Step 2: Preprocess the text
            text_content = preprocess_text(text_output_file)

            # Step 3: Chunk the text with overlap
            # Larger chunk length means faster processing, but context fetched for RAG will be larger and thus less fine-grained
            max_chunk_length = 1000  # Maximum number of words per chunk
            overlap_length = 250     # Number of words to overlap between chunks
            chunks = chunk_text(text_content, max_length=max_chunk_length, overlap=overlap_length)
            print(f'Total chunks created: {len(chunks)}.')

            total_tokens = len(chunks) * (max_chunk_length - overlap_length)
            print(f"Estimated total tokens: {total_tokens:,}.")

            estimated_cost = (total_tokens / 1_000_000) * 0.1
            print(f"Estimated cost for embeddings: ${estimated_cost:.5f}.")

            processing_rate = 3  # chunks per second; varies by usage tier (https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-five)
            total_time_seconds = len(chunks) / processing_rate

            hours = int(total_time_seconds // 3600)
            minutes = int((total_time_seconds % 3600) // 60)
            seconds = int(total_time_seconds % 60)

            print(f"Total estimated processing time: {hours}h {minutes}m {seconds}s")

            continue_processing = input("Would you like to continue with embedding generation? (y/n): ").strip().lower()
            if continue_processing != 'y':
                print("Exiting the program.")
                exit()

            # Step 4: Generate embeddings
            embeddings = generate_embeddings(chunks)

            # Check if FAISS index exists
            index_exists = os.path.exists('faiss_index.idx') and os.path.exists('chunks.pkl')

            if index_exists:
                action = input("A FAISS index was found. Would you like to add to it or reset it? (add/reset): ").strip().lower()
                while action not in ['add', 'reset']:
                    action = input("Please enter 'add' to use the existing index or 'reset' to recreate it: ").strip().lower()
                if action == 'add':
                    existing_index, existing_chunks = load_index_and_chunks()
                    if existing_index is not None and existing_chunks is not None:
                        existing_index.add(np.array(embeddings).astype('float32'))
                        existing_chunks.extend(chunks)
                        save_index_and_chunks(existing_index, existing_chunks)
                        index = existing_index
                        chunks = existing_chunks
                    else:
                        print("Failed to load existing FAISS index. Creating a new one.")
                        index = create_faiss_index(embeddings)
                        save_index_and_chunks(index, chunks)
                else:
                    index = create_faiss_index(embeddings)
                    save_index_and_chunks(index, chunks)
            else:
                # Step 5: Create FAISS index
                index = create_faiss_index(embeddings)

                # Step 6: Save index and chunks
                save_index_and_chunks(index, chunks)
        else:
            # Load existing FAISS index and chunks
            index, chunks = load_index_and_chunks()
            print("Loaded existing FAISS index and chunks.")
    else:
        print("Running in normal mode. No RAG components will be used.")

    # Step 7: Handle query
    # Replace the user input with reading from a text file
    with open('query.txt', 'r', encoding='utf-8') as file:
        user_query = file.read().strip()

    if run_mode == 'rag' and index and chunks:
        # RAG mode: Retrieve relevant chunks for a query
        relevant_chunks = retrieve_chunks(user_query, index, chunks)
        print("\nRetrieved chunks.")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\nChunk {i + 1}:\n{chunk}")

        # Step 8: Generate answer using relevant chunks (RAG)
        answer = generate_answer(user_query, relevant_chunks)
    else:
        # Normal mode: Generate answer directly without RAG
        answer = normal_chat_mode(user_query)
    
    try:
        with open('response.txt', 'w', encoding='utf-8') as response_file:
            response_file.write(answer)
        print("\nAnswer has been written to 'response.txt'.")
    except Exception as e:
        print(f"Failed to write the answer to 'response.txt': {e}")

    print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
