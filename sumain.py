# SUMAIN.PY
import os
from reader import Reader
from adapter import Adapter
from embedder import Embedder
from vector_db import VectorDatabase
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define your prompt template
prompt_template = """Summarize the context to answer the question. 

Context: {context}

Question: {question}

Answer:"""

document_path = "/home/praadnyah/LLM/miniproj/RAG/PDF/cancer.pdf"

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
    llm_query = "What is cervical cancer?"
    llm_query_embedding = doc_embedder.embed_query(llm_query)

    # Perform a search in the vector database
    search_results = vdb.search_query(llm_query_embedding, chunk_count=1)
    
    # Ensure search results return the correct number of values
    if len(search_results) != 1:
        raise ValueError("Search query did not return the expected number of values")

    search_chunks = search_results

    # Print search results
    print("Search Results as context:\n")
    for i, (chunk, similarity) in enumerate(search_chunks):
        print(f"Result {i+1}: {chunk}\n, Similarity: {similarity}\n")

    # Load DistilBART model and tokenizer
    model_name = "lxyuan/distilbart-finetuned-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Combine the search results as context
    context = " ".join([chunk for chunk, _ in search_chunks])

    prompt = prompt_template.format(context=context, question=llm_query)

    # Tokenize input and generate answer with DistilBART
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    # Generate answer ensuring it completes the sentence
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=150,  # Increase max_new_tokens as needed
        eos_token_id=tokenizer.eos_token_id,  # Ensure sentences are complete
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Generated Answer: {generated_answer}")

except Exception as e:
    print(f"An error occurred: {e}")
