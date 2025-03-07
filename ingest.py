import os  # this module is used to interact with the operating system
import time  # this module is used to handle time-related operations
from dotenv import (
    load_dotenv,
)  # this module is used to load environment variables from a .env file
from langchain_community.document_loaders import (
    PyPDFLoader,
)  # this module is used to load documents from a PDF file
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)  # this module is used to split documents into chunks
from langchain_chroma import (
    Chroma,
)  # this module is used to create a vector store in a sqlite database
from uuid import (
    uuid4,
)  # this module is used to generate unique identifiers that are used to identify the documents in the vector store
from models import (
    Models,
)  # this module is used to load the models from the models.py file
from langchain_core.documents import (
    Document,
)  # this module is used to create a document

load_dotenv()  # this function is used to load the environment variables from the .env file


models = Models()  # this function is used to load the models from the models.py file
embeddings = (
    models.embeddings_ollama
)  # this function is used to load the embeddings from the models.py file
llm = (
    models.chat_ollama
)  # this function is used to load the chat model from the models.py file

# Define constants
data_folder = "C:/Users/Bas/Downloads/rag/data"  # this is the path to the folder where the documents are stored
chunk_size = (
    1000  # this is the size of the chunks that will be used to split the documents
)
chunk_overlap = 200  # this is the overlap between the chunks
check_interval = 10  # this is the interval at which the program will check for new documents in the data folder

# The following code is used to create a vector store in a sqlite database
vector_store = Chroma(
    collection_name="documents",  # this is the name of the collection that will be used to store the documents in the vector store
    embedding_function=embeddings,  # this is the embedding function that will be used to embed the documents
    persist_directory="./db/chroma_langchain_db",  # this is the path to the folder where the vector store will be stored
)


# This function is used to ingest documents into the vector store
def ingest_file(file_path):
    # Skip non-pdf_files
    if not file_path.lower().endswith(
        ".pdf"
    ):  # this is used to check if the file is a PDF file
        print(f"Skipping non-pdf file: {file_path}")
        return

    # Load documents
    loader = PyPDFLoader(
        file_path
    )  # this is used to load the PDF file into a list of documents
    loaded_documents = (
        loader.load()
    )  # this is used to load the documents from the PDF file
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", " ", ""]
    )  # this is used to split the documents into chunks. The separators are used to split the documents into chunks.
    chunks = text_splitter.split_documents(
        loaded_documents
    )  # chunks contains the content of the split documents

    print(
        f"Processing file: {os.path.basename(file_path)}"
    )  # this is used to print the name of the file that is being processed to the console
    print(
        f"Document split into {len(chunks)} chunks"
    )  # this is used to print the number of chunks that the document has been split into to the console

    # Process chunks to remove problematic characters
    processed_chunks = []  # this is a predefined list used to store the processed chunks
    for i, chunk in enumerate(chunks, 1):  # this is used to iterate over the chunks
        # Access the page_content attribute
        if hasattr(
            chunk, "page_content"
        ):  # this is used to check if the chunk has a page_content attribute
            text = chunk.page_content  # this is used to get the content of the chunk
            # Clean the text to remove problematic characters
            clean_text = text.encode("utf-8", "ignore").decode(
                "utf-8"
            )  # Clean the text to remove problematic characters
            # Create a new Document with the cleaned text and original metadata
            metadata = (
                chunk.metadata if hasattr(chunk, "metadata") else {}
            )  # this is used to get the metadata of the chunk
            processed_chunk = Document(
                page_content=clean_text, metadata=metadata
            )  # this is used to create a new Document with the cleaned text and original metadata
            processed_chunks.append(
                processed_chunk
            )  # this is used to append the processed chunk to the processed_chunks list

            # Print progress every 10 chunks or at the end
            if i % 10 == 0 or i == len(
                chunks
            ):  # this is used to print the progress of the processing of the chunks
                print(
                    f"Progress: {i}/{len(chunks)} chunks processed ({(i / len(chunks) * 100):.1f}%)"
                )  # this is used to print the progress of the processing of the chunks

    uuids = [
        str(uuid4()) for _ in range(len(processed_chunks))
    ]  # this is used to generate unique identifiers for the documents
    print(
        f"Ingesting {len(processed_chunks)} chunks into vector store..."
    )  # this is used to print the number of chunks that are being ingested into the vector store
    vector_store.add_documents(
        documents=processed_chunks, ids=uuids
    )  # this is used to add the processed chunks to the vector store
    print(
        f"Successfully ingested {len(processed_chunks)} chunks from {os.path.basename(file_path)}"
    )  # this is used to print the number of chunks that have been ingested from the file


# Main loop
def main_loop():
    print(
        f"Starting document ingestion service. Monitoring folder: {data_folder}"
    )  # this is used to print the folder that is being monitored to the console
    print(
        f"Documents will be checked every {check_interval} seconds."
    )  # this is used to print the interval at which the program will check for new documents in the data folder to the console
    print(
        "Add PDF files to this folder to ingest them into the vector database."
    )  # this is used to print the message to the console
    print("=" * 80)  # this is used to print a line of dashes to the console

    while True:
        # Get all files in the data folder
        files = os.listdir(
            data_folder
        )  # this is used to get all the files in the data folder
        pdf_files = [
            f for f in files if f.endswith(".pdf") and not f.startswith("_")
        ]  # this is used to get all the PDF files in the data folder
        pdf_count = len(
            pdf_files
        )  # this is used to get the number of PDF files in the data folder

        if (
            pdf_count > 0
        ):  # this is used to check if there are any PDF files in the data folder
            print(
                f"Found {pdf_count} PDF files to process in {data_folder}"
            )  # this is used to print the number of PDF files that have been found in the data folder to the console

            processed_count = 0  # this is used to count the number of PDF files that have been processed
            for filename in pdf_files:  # this is used to iterate over the PDF files
                file_path = os.path.join(
                    data_folder, filename
                )  # this is used to get the path to the PDF file
                ingest_file(
                    file_path
                )  # this is used to ingest the PDF file into the vector store
                new_filename = "_" + filename  # this is used to rename the PDF file
                new_file_path = os.path.join(
                    data_folder, new_filename
                )  # this is used to get the path to the new PDF file
                os.rename(
                    file_path, new_file_path
                )  # this is used to rename the PDF file
                processed_count += 1  # this is used to increment the processed count

            print(
                f"Processed {processed_count} PDF files."
            )  # this is used to print the number of PDF files that have been processed to the console
            print("=" * 80)  # this is used to print a line of dashes to the console
            print(
                f"All documents have been processed successfully!"
            )  # this is used to print the message to the console
            print(
                f"Processed documents are prefixed with '_' and remain in the data folder."
            )  # this is used to print the message to the console
            print(
                f"Waiting for new documents to be added to {data_folder}"
            )  # this is used to print the message to the console
            print(
                f"Next check in {check_interval} seconds..."
            )  # this is used to print the message to the console
            print("=" * 80)  # this is used to print a line of dashes to the console
        else:
            # Only print the waiting message if there were no files to process
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No new documents found. Waiting for documents to be added to {data_folder}"
            )  # this is used to print the message to the console
            print(
                f"Add PDF files to {data_folder} for processing. Next check in {check_interval} seconds..."
            )  # this is used to print the message to the console
            print("=" * 80)  # this is used to print a line of dashes to the console

        time.sleep(
            check_interval
        )  # this is used to sleep for the interval at which the program will check for new documents in the data folder


if __name__ == "__main__":
    main_loop()  # this is used to call the main_loop function
