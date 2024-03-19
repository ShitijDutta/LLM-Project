import os
import logging
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse

load_dotenv()
llamaparse_key = os.getenv("LLAMA_CLOUD_API_KEY")

MARKDOWN_FILE_PATH = "converted_files/"
LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "converter.log")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize logging
logging.basicConfig(filename=LOG_FILE, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main(): 
    logger.info("Starting main function.")
    pdf_to_md()
    logger.info("Main function completed.")

def pdf_to_md():
    logger.info("Starting conversion.")
    documents = converter()
    convert_to_markdown(documents)
    logger.info("Document conversion complete.")


def converter():
    llamaparser = LlamaParse(
        api_key=llamaparse_key,
        result_type="markdown",
        verbose=True
    )
    file_extractor = {".pdf": llamaparser}
    documents = SimpleDirectoryReader(input_dir='uploads', file_extractor=file_extractor).load_data()
    logger.info("Documents loaded after being parsed from LlamaParse")

    for document in documents:
        logger.info(f"Document loaded: {document.text[:100]}")
    
    return documents

def convert_to_markdown(documents):
    for document in documents:
        # Extract the text from the document
        text_content = document.text

        # Create a unique filename based on the document ID
        filename = f"{MARKDOWN_FILE_PATH}document_{document.id_}.txt"

        # Check if the file already exists
        if not os.path.exists(filename):
            # Save the text data to a Markdown file with UTF-8 encoding
            with open(filename, 'w', encoding='utf-8') as markdown_file:
                markdown_file.write(text_content)
            logger.info(f"Text data for Document {document.id_} successfully saved to {filename}")
        else:
            logger.warning(f"File {filename} already exists. Skipped creating a new file.")

if __name__ == "__main__":
    logger.info("Executing markdown script.")
    main()
    logger.info("Markdown script execution complete.")
