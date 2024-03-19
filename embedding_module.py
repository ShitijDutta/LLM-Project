import uuid
import os
import logging
from dotenv import load_dotenv
from multiprocessing import Pool

import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions

from langchain.indexes import SQLRecordManager, index
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.schema import Document


load_dotenv()

LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "app.log")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize logging
logging.basicConfig(filename=LOG_FILE, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


key = os.getenv("OPENAI_API_KEY")


headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4")
]


CHROMA_PATH = "chroma"
UPLOADS_FILE_PATH = "uploads"
MARKDOWN_FILE_PATH = "C:/Users/shitij/Desktop/RAG QT/converted_files"

client = chromadb.HttpClient(host='localhost', settings=Settings(anonymized_telemetry=False), port=8000)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=key,
                model_name='text-embedding-3-small'
            )

collection = client.get_or_create_collection(name='my-collections-markdown', embedding_function=openai_ef)

def main():
    logger.info("Starting main function.")
    generate_data_store()
    logger.info("Main function completed.")


def generate_data_store():
    logger.info("Generating data store.")
    documents = load_markdown(MARKDOWN_FILE_PATH)
    split_text(documents)
    logger.info("Data store generation complete.")

def load_markdown(directory):
    logger.info(f"Loading documents from directory: {directory}")
    loader_mapping = {
        '.txt': TextLoader
    }
    
    documents = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        file_extension = os.path.splitext(file)[1]
        
        if file_extension in loader_mapping:
            loader = loader_mapping[file_extension](file_path, encoding='utf-8')
            documents.extend(loader.load())
            print(type(documents))
    return documents

def add_chunks_to_collection(chunk):
    collection.add(ids=[str(uuid.uuid1())], documents=chunk.page_content, metadatas=chunk.metadata)

def split_text(documents: list[Document]):
    logger.info("Splitting text")
    text_splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_documents(documents)

    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    logger.info("Saving to chroma")

    with Pool(processes=8) as pool:
        pool.map(add_chunks_to_collection, chunks)

    logger.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    logger.info("Executing embedding_module script.")
    main()
    logger.info("Embedding_module script execution complete.")
