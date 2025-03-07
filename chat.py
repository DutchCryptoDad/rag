from langchain_core.prompts import ChatPromptTemplate # this module is used to create a prompt template
from langchain.chains.combine_documents import create_stuff_documents_chain # this module is used to create a chain that combines documents
from langchain.chains.retrieval import create_retrieval_chain # this module is used to create a chain that retrieves documents
from langchain_chroma import Chroma # this module is used to create a vector store in a sqlite database
from models import Models # this module is used to load the models from the models.py file
import threading # this module is used to handle threading
import time # this module is used to handle time
import sys # this module is used to handle system-related operations
import itertools # this module is used to handle itertools

# load_dotenv()

models = Models() # this function is used to load the models from the models.py file
embeddings = models.embeddings_ollama # this function is used to load the embeddings from the models.py file
llm = models.chat_ollama # this function is used to load the chat model from the models.py file

# Define constants
vector_store = Chroma(
    collection_name="documents", # this is the name of the collection that will be used to store the documents in the vector store
    embedding_function=embeddings, # this is the embedding function that will be used to embed the documents
    persist_directory="./db/chroma_langchain_db" # this is the path to the folder where the vector store will be stored
)

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions about the documents in the database."), # this is the system prompt that will be used to answer the questions
    ("user", "Use the user question {input} to answer the question. Use only the {context} to answer the question. If you don't know the answer, say 'I don't know'"), # this is the user prompt that will be used to answer the questions
])

# Define the chain
retriever = vector_store.as_retriever(kwargs={"k": 5}) # this is the retriever that will be used to retrieve the documents from the vector store
combine_docs_chain = create_stuff_documents_chain(llm, prompt) # this is the chain that will be used to combine the documents

# Create the retrieval chain
retrieval_chain = create_retrieval_chain(
    retriever, combine_docs_chain # this is the retrieval chain that will be used to retrieve the documents from the vector store
)

def run_query(query, result_container):
    """
    Run the query in a separate thread and store the result in the provided container.
    
    Args:
        query (str): The user's query
        result_container (list): A list to store the result
    """
    result = retrieval_chain.invoke({"input": query}) # this is the result that will be used to answer the questions
    result_container.append(result) # this is the result that will be used to answer the questions

def spinner_animation():
    """
    Generator function for a simple spinner animation.
    """
    spinner = itertools.cycle(['-', '/', '|', '\\']) # this is the spinner that will be used to display a progress bar while waiting for the result
    while True:
        yield next(spinner)

# Main loop
def main():
    while True:
        user_input = input("Enter a question (or type 'q' to quit): ") # this is the user input that will be used to answer the questions
        if user_input.lower() == 'q':
            print("Exiting...") # this is the message that will be printed to the console when the user quits the program
            break
        
        # Create a container for the result
        result_container = [] # this is the container that will be used to store the result
        
        # Start the query in a separate thread
        query_thread = threading.Thread(
            target=run_query, # this is the target that will be used to run the query
            args=(user_input, result_container) # this is the arguments that will be used to run the query
        )
        query_thread.start() # this is the function that will be used to start the query
        
        # Display a spinner while waiting for the result
        spinner = spinner_animation() # this is the spinner that will be used to display a progress bar while waiting for the result
        print("Ollama is thinking ", end="", flush=True) # this is the message that will be printed to the console while waiting for the result
        while query_thread.is_alive():
            sys.stdout.write(next(spinner)) # this is the message that will be printed to the console while waiting for the result
            sys.stdout.flush() # this is the message that will be printed to the console while waiting for the result
            time.sleep(0.1) # this is the message that will be printed to the console while waiting for the result
            sys.stdout.write('\b') # this is the message that will be printed to the console while waiting for the result
        
        print("\n")  # Clear the spinner line
        
        # Get the result from the container
        result = result_container[0] # this is the result that will be used to answer the questions
        print(result["answer"]) # this is the message that will be printed to the console when the result is retrieved
        
        # Print source documents if available
        if "context" in result:
            print("\nSources:") # this is the message that will be printed to the console when the sources are retrieved
            for i, doc in enumerate(result["context"]):
                print(f"Source {i+1}:") # this is the message that will be printed to the console when the sources are retrieved
                print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content) # this is the message that will be printed to the console when the sources are retrieved
                print() # this is the message that will be printed to the console when the sources are retrieved

if __name__ == "__main__":
    main() # this is the function that will be used to start the program
