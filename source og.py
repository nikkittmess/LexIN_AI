import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time


# Initialize legal sources
LEGAL_RESOURCES = {
    'web_sources': ['https://indiankanoon.org/', 'https://inshorts.com/en/read'],
    'pdf_paths': "G:\CODES\CHATBOT\CHATBOT\IC.pdf"
}


def scrape_legal_sites(query):
    scraped_text = ""

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Start Chrome browser
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    for url in LEGAL_RESOURCES['web_sources']:
        try:
            if "indiankanoon" in url:
                search_url = f"{url}search/?formInput={query.replace(' ', '+')}"
                driver.get(search_url)
            else:
                driver.get(url)

            time.sleep(2)  # wait for page to load
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            if "inshorts" in url:
                cards = soup.find_all('div', class_='news-card-content')
                for card in cards[:5]:
                    scraped_text += card.get_text(strip=True) + "\n"
            else:
                paragraphs = soup.find_all('p')
                for p in paragraphs[:10]:
                    scraped_text += p.get_text(strip=True) + "\n"

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    driver.quit()
    return scraped_text

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:                          # Open PDF in binary read mode
        reader = PdfReader(file)                          # Create a PDF reader object
        text = ''
        for page in reader.pages:                                # Loop through each page
            text += page.extract_text() or ''                    # Extract and append text (ignore None)
    return text                                                  # Return the full extracted text

# Load pre-trained QA model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"          # Choose a lightweight QA model
tokenizer = AutoTokenizer.from_pretrained(model_name)           # Load the tokenizer for the model
model = AutoModelForQuestionAnswering.from_pretrained(model_name)  # Load the actual QA model

# Extract context from the PDF file
pdf_context = extract_text_from_pdf(LEGAL_RESOURCES['pdf_paths'])  # Call the function to get text from PDF

# Function to answer user questions
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)  # Tokenize inputs
    with torch.no_grad():                                          # No gradient calculation needed
        outputs = model(**inputs)                                  # Run model on inputs
    answer_start = torch.argmax(outputs.start_logits)              # Get the start position of answer
    answer_end = torch.argmax(outputs.end_logits) + 1              # Get the end position of answer
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )                                                              # Decode tokens into answer text
    return answer                                                  # Return the final answer

# Example question
user_question = "What is Article 21 of the Indian Constitution?"
print("Answer:", answer_question(user_question, pdf_context))     # Print the answer from PDF context
