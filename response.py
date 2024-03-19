from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
import os
import chromadb
import logging
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "app.log")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma"
client = chromadb.HttpClient(host='localhost', settings=Settings(anonymized_telemetry=False), port=8000)
embedding_function = OpenAIEmbeddings(model='text-embedding-3-small', api_key=key)
db = Chroma(client=client, collection_name='my-collections-markdown', embedding_function=embedding_function, client_settings=Settings(anonymized_telemetry=False))
retriever = db.as_retriever(search_kwargs={"k":10})

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | model
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)