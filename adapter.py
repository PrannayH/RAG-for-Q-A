from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Adapter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 30, verbose: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verboseprint = print if verbose else lambda *a: None

        self.verboseprint(
            f"ADAPTER: Adapter initialised successfully with the following configuration: chunk_size = {self.chunk_size}  chunk_overlap = {self.chunk_overlap}"
        )

    def _document_dictionary_to_lang_chain_document(self, text: str) -> Document:
        return Document(page_content=text, metadata={})

    def get_chunks(self, docs: list[str]):
        chunk_list = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separators=["\n\n", ".", "?", "!", " "]
        )
        
        for d in docs:
            document = self._document_dictionary_to_lang_chain_document(d)
            texts = text_splitter.split_documents([document])

            for text in texts:
                chunk_list.append(text.page_content)

        self.verboseprint(f"ADAPTER: Document chunking successful. Number of chunks = {len(chunk_list)}")
        return chunk_list
