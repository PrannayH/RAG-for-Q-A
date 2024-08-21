import os
import torch
from reader import Reader
from adapter import Adapter
from embedder import Embedder
from vector_db import VectorDatabase
from transformers import BertForQuestionAnswering, BertTokenizer

# Define your prompt template
prompt_template = """Answer using the context in 4 lines. 

Context: {context}

Question: {question}

Answer:"""

document_path = "/home/praadnyah/LLM/miniproj/RAG/PDF/diabetes.pdf"

try:
    # Initialize the reader and load the document
    doc_reader = Reader(document_type="pdf", verbose=True)
    texts, document_name, document_type = doc_reader.load_document(document_path)
    
    print(f"Document Name: {document_name}")
    print(f"Document Type: {document_type}")
    print(f"Number of pages loaded: {len(texts)}")

    # Combine text into documents for embedding
    adapter = Adapter(verbose=True)
    chunks = adapter.get_chunks(texts)

    # Initialize the embedder and get embeddings for the documents
    doc_embedder = Embedder(embedding_model="all-MiniLM-l6-v2", verbose=True)
    embeddings = doc_embedder.embed_documents(chunks)

    # Initialize the vector database and insert embeddings
    vdb = VectorDatabase(verbose=True)
    success = vdb.insert_embeddings(embeddings, chunks)

    if success:
        print("Embeddings inserted successfully into the vector database.")
    else:
        print("Failed to insert embeddings into the vector database.")

    # Example usage for query embedding
    llm_query = "What are symptoms of diabetes?"
    llm_query_embedding = doc_embedder.embed_query(llm_query)

    # Perform a search in the vector database
    search_results = vdb.search_query(llm_query_embedding, chunk_count=2)
    
    # Ensure search results return the correct number of values
    if len(search_results) != 2:
        raise ValueError("Search query did not return the expected number of values")

    search_chunks, _ = search_results

    # Print search results
    print("Search Results as context:\n")
    for i, chunk in enumerate(search_chunks):
        print(f"Result {i+1}: {chunk}\n")

    # Load Roberta model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

     # Combine the search results as context
    context = " ".join(search_chunks)
    # context = search_chunks[0]
    

    prompt = prompt_template.format(context=context, question=llm_query)

    # Tokenize input and generate answer with BERT
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Get the most likely answer
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits) + 1
    generated_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())[answer_start:answer_end])

    # Ensure the answer ends with a complete sentence
    sentences = generated_answer.split(".")
    if len(sentences) > 1:
        generated_answer = ".".join(sentences[:-1]) + "."

    print(f"Generated Answer: {generated_answer}")
   

except Exception as e:
    print(f"An error occurred: {e}")
