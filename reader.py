import os
from PyPDF2 import PdfReader

class Reader:
    def __init__(self, document_type: str = "pdf", verbose: bool = True):
        self.verboseprint = print if verbose else lambda *a: None
        self.document_type = document_type
        self.current_directory = os.getcwd()
        self.verboseprint("READER: Reader initialised successfully")

    def extract_paragraphs(self, text):
        """Extracts paragraphs from the extracted text."""
        paragraphs = []
        temp_paragraph = []
        for line in text.splitlines():
            line = line.strip()
            if line:  # Non-empty line
                temp_paragraph.append(line)
            elif temp_paragraph:  # Empty line indicates end of paragraph
                paragraphs.append(" ".join(temp_paragraph))
                temp_paragraph = []
        
        # Append the last paragraph if not already added
        if temp_paragraph:
            paragraphs.append(" ".join(temp_paragraph))
        
        return paragraphs

    def load_document(self, document_location: str) -> tuple:
        """Loads the PDF document and extracts paragraph text."""
        
        paragraphs = []

        try:
            with open(document_location, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        page_paragraphs = self.extract_paragraphs(page_text)
                        paragraphs.extend(page_paragraphs)

            document_name = os.path.basename(document_location)
            document_type = self.document_type

            return paragraphs, document_name, document_type

        except FileNotFoundError:
            raise ValueError(f"File not found at location: {document_location}")
        except Exception as e:
            raise ValueError(f"An error occurred while reading the document: {e}")
        