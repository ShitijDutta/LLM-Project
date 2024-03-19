from typing import List
import chromadb
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from langchain_community.document_transformers import LongContextReorder
from chromadb.config import Settings
from langchain.retrievers.multi_query import MultiQueryRetriever

from dotenv import load_dotenv
load_dotenv()
import os
import logging

key = os.getenv("OPENAI_API_KEY")

LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "app.log")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma"

client = chromadb.HttpClient(host='localhost', port=8000, settings=Settings(anonymized_telemetry=False))
embedding_function = OpenAIEmbeddings(model='text-embedding-3-small', api_key=key)

db = Chroma(client=client, collection_name='my-collections-markdown', embedding_function=embedding_function, persist_directory="CHROMA_PATH")

retriever = db.as_retriever(search_type="mmr")

# Define Pydantic BaseModel for output parsing
class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")

# Define Pydantic Output Parser
class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

# Function to perform Cross-encoder re-ranking
def cross_encoder_rerank(unique_contents, query):
    logging.info("Performing Cross-encoder re-ranking...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc] for doc in unique_contents]
    scores = cross_encoder.predict(pairs)
    scored_docs = zip(scores, unique_contents)
    sorted_docs = sorted(scored_docs, reverse=True)
    reranked_docs = [doc for _, doc in sorted_docs][:8]
    logging.info("Cross-encoder re-ranking completed.")
    return reranked_docs

# Function to perform Long Context Reorder
def long_context_reorder(reranked_docs):
    logging.info("Performing Long Context Reorder...")
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(reranked_docs)
    logging.info("Long Context Reorder completed.")
    return reordered_docs

# Main function to execute all steps
def execute_pipeline(query: str) -> str:
    if not query:
        logging.warning("Empty query provided. Skipping pipeline execution")
        return ""
    
    logging.info("Starting pipeline execution...")
    # Initializing the LLM
    llm = ChatOpenAI(temperature=0)

    # Define the prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template='You are an AI language model assistant. Your task is to generate three'
        'different versions of the given user question to retrieve relevant documents from a vector'
        'database. By generating multiple perspectives on the user question, your goal is to help'
        'the user overcome some of the limitations of the distance-based similarity search.'
        'Provide these alternative questions separated by newlines. Only provide the query, no numbering.'
        'Original question: {question}'
    )

    # Initialize LLMChain
    llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=LineListOutputParser())
    
    # Perform document retrieval
    logging.info("Performing document retrieval...")
    final_retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=llm_chain, parser_key="lines"
    )

    unique_docs = final_retriever.get_relevant_documents(query=query)
    
    logging.info("Retrieved %s unique documents.", len(unique_docs))

    # Cross-encoder re-ranking
    doc_texts = [doc.page_content for doc in unique_docs]
    reranked_docs = cross_encoder_rerank(doc_texts, query)
    logging.info("Cross-encoder re-ranked documents: %s", reranked_docs)

    # Long Context Reorder
    reordered_docs = long_context_reorder(reranked_docs)
    reordered = "\n\n".join(reordered_docs)
    
    logging.info("Reordered documents: %s", reordered)

    logging.info("Pipeline execution completed.")

    return reordered
