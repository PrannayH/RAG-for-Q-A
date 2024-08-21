# EMBEDDER.PY
from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:
    def __init__(self, embedding_model: str = "all-MiniLM-l6-v2", verbose: bool = True):
        self.model_name = embedding_model
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.verboseprint = print if verbose else lambda *a: None

        self.verboseprint(
            f"EMBEDDER: Embedder initialised successfully with configuration: embedding_model = {self.model_name}"
        )

    def embed_documents(self, documents: list[str]):
        """Generates the embedding for each document in the list"""
        embedding_list = self.embedding_model.embed_documents(documents)

        self.verboseprint(
            f"EMBEDDER: Embeddings generated successfully. Number of output = {len(embedding_list)}"
        )

        return embedding_list

    def embed_query(self, query: str):
        """Generates the embedding for the given query"""
        embedding_vector = self.embedding_model.embed_query(query)
        return embedding_vector
