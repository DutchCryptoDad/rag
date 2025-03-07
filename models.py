from langchain_ollama import OllamaEmbeddings, ChatOllama


class Models:
    def __init__(self):
        # ollama pull mxbai-embed-text
        self.embeddings_ollama = OllamaEmbeddings(
            model="mxbai-embed-large"
        )  # this is the embeddings model that will be used to embed the documents
        # ollama pull llama3.2
        self.chat_ollama = ChatOllama(
            model="llama3.2", temperature=0.0
        )  # this is the chat model that will be used to answer the questions
