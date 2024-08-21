import numpy as np
import faiss

class VectorDatabase:
    def __init__(self, name='faiss', verbose=False):
        self.name = name
        self.verbose = verbose
        self.index = None
        self.documents = []

    def _initialize_faiss_index(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def insert_embeddings(self, embeddings, documents):
        if self.verbose:
            print(f"VECTOR DATABASE: Initializing {self.name} with configuration: name = {self.name}")
        
        # Convert embeddings to NumPy array
        embeddings = np.array(embeddings)
        
        # Normalize embeddings to unit length
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        dimension = embeddings.shape[1]
        self._initialize_faiss_index(dimension)
        
        if self.verbose:
            print(f"VECTOR DATABASE: Inserting {len(embeddings)} embeddings into the database.")

        self.index.add(embeddings)
        self.documents.extend(documents)

        return True

    def search_query(self, query_embedding, chunk_count=3):
        if self.index is None:
            raise ValueError("The FAISS index has not been initialized or embeddings have not been inserted.")
        
        if self.verbose:
            print(f"VECTOR DATABASE: Performing search for the query embedding.")

        # Normalize query embedding
        query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
        
        # Perform search with normalized query embedding
        distances, indices = self.index.search(np.array([query_embedding_normalized]), chunk_count)

        if self.verbose:
            print(f"VECTOR DATABASE: Search successful. Found {len(indices[0])} results.")

        # Ensure indices are integers for reconstruction
        indexed_embeddings = np.array([self.index.reconstruct(int(idx)) for idx in indices[0]])
        indexed_embeddings_normalized = indexed_embeddings / np.linalg.norm(indexed_embeddings, axis=1, keepdims=True)
        cosine_similarities = np.dot(indexed_embeddings_normalized, query_embedding_normalized)

        # Return search results with cosine similarities
        search_chunks = [(self.documents[i], cosine_similarities[idx]) for idx, i in enumerate(indices[0])]
        
        if len(search_chunks) < chunk_count:
            print(f"WARNING: Search returned only {len(search_chunks)} results instead of {chunk_count}.")
        
        return search_chunks
