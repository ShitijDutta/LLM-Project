import os
from pathlib import Path
import streamlit as st
import pandas as pd
import logging
from dotenv import load_dotenv

from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from embedding_module import generate_data_store
from response import rag_chain_with_source
from markdown import pdf_to_md
from qt_response import execute_pipeline

load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = 'default'


LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "app.log")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize logging
logging.basicConfig(filename=LOG_FILE, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


UPLOAD_FOLDER = 'uploads'
RECORD_FILE = 'uploaded_files.txt'

key = os.getenv("OPENAI_API_KEY")


def open_file(file_path):
    try:
        os.startfile(file_path)
    except OSError as e:
        st.error(f"Failed to open the file'{file_path}':{e}")

# Function to handle file upload and processing
def upload_files(upload_files):
    uploaded_file_names = [file.name for file in upload_files]

    # Check if record file exists
    if os.path.exists(RECORD_FILE):
        with open(RECORD_FILE, 'r') as f:
            existing_files = f.read().splitlines()
            duplicate_files = set(uploaded_file_names) & set(existing_files)
            if duplicate_files:
                st.warning(f"The following files already exist in the vector store: {', '.join(duplicate_files)}")
                return

    # Process files (chunking, embedding, vector store creation)
    st.info("Uploading files...")

    save_directory = Path(UPLOAD_FOLDER)
    save_directory.mkdir(parents=True, exist_ok=True)

    for uploaded_file in upload_files:
        # Save the file to the specified directory
        file_path = save_directory / uploaded_file.name
        with open(file_path, "wb") as file:
            file.write(uploaded_file.read())

    st.success("File Upload Complete")


def generate_response(question, context):
    prompt = ChatPromptTemplate.from_template('Answer the question based only on the following CONTEXT:\n' + context + '\nQuestion: {question}'
                                              'Answer the QUESTION using the CONTEXT text. '
                                              'Keep your answer ground in the facts of the CONTEXT.'
                                              'If the DOCUMENT doesnt contain the facts to answer the QUESTION say That you cannot answer the question'
                                            )
    
    output_parser = StrOutputParser()
    model = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
    chain = (
        {
            "question": RunnablePassthrough()
        } 
        | prompt
        | model
        | output_parser
    )

    result = chain.invoke(question)
    return result


# Streamlit app
def main():
    result = ""
    final = ""
    st.title('Retrieval Augmented Generation')
    st.markdown("---")

    # Sidebar with file upload section and uploaded files record
    st.sidebar.subheader("Upload Files")
    uploaded_files = st.sidebar.file_uploader("Choose files to upload", type=['txt', 'pdf', 'xlsx'], accept_multiple_files=True)
    if uploaded_files:
        upload_files(uploaded_files)

    # Update file upload dataframe
    st.sidebar.subheader("Uploaded Files Record")
    
    col1, col2 = st.columns([3, 1], gap="medium")
    with col1:
        # Button to trigger PDF file conversion
        if st.button("Convert"):
            st.info("Converting Uploaded PDFs to Markdown format")
            pdf_to_md()
            st.success("File Conversion complete!")  
    
    with col2:
    # Button to trigger data store generation
        if st.button("Generate Data Store"):
            generate_data_store()

            files = os.listdir(UPLOAD_FOLDER)
            with open(RECORD_FILE, 'w') as f:
                    for file_name in files:
                        f.write(file_name + '\n')

            if os.path.exists(RECORD_FILE):
                uploaded_files_df = pd.read_csv(RECORD_FILE, header=None, names=["File"])
                st.sidebar.dataframe(uploaded_files_df)

    # Main component with chat interface
    st.subheader("Chat with Documents")
    user_question = st.text_input("Enter your question:")
    response_text = st.empty()
    col3, col4, = st.columns([7.5,1], gap="medium")
    with col3:
        if st.button("Get Answer"):
            # Invoke QA
            result = rag_chain_with_source.invoke(user_question)
            response = result['answer']
            # Display the result
            response_text.text_area("RESPONSE:", response)
            source_path = result['context'][0].metadata.get('source', '')
            source_path_normalized = os.path.normpath(source_path)
            if st.button("Open Source File"):
                open_file(source_path_normalized)
            print("File path:", source_path_normalized)

    with col4:
        if st.button("Get QT"):
            context = execute_pipeline(query=user_question)
            final = generate_response(question=user_question, context=context)
            response_text.text_area("RESPONSE:", final)
        

# Run the app
if __name__ == "__main__":
    main()
