import streamlit as st
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

# Initialize Streamlit
st.title("Medical Document Q&A with RAG")

# Step 1: Accept a PDF document
uploaded_file = st.file_uploader("Upload a PDF document (Max size: 1MB)", type="pdf")
chunk_count = st.number_input("Enter the number of chunks to retrieve:", min_value=1, max_value=10, value=2)
llm_query = st.text_input("Enter your query:")

if uploaded_file and llm_query and chunk_count:
    
    try:
        # Placeholder for logging messages
        log_placeholder = st.empty()

        # Save the uploaded file
        document_path = os.path.join("/tmp", uploaded_file.name)
        with open(document_path, "wb") as f:
            f.write(uploaded_file.read())

        # Step 2: Initialize the reader and load the document
        log_placeholder.text("Initializing reader...")
        doc_reader = Reader(document_type="pdf", verbose=True)
        texts, document_name, document_type = doc_reader.load_document(document_path)
        log_placeholder.text(f"Document Name: {document_name}, Document Type: {document_type}, Number of pages loaded: {len(texts)}")

        # Step 3: Combine text into documents for embedding
        log_placeholder.text("Chunking completed...")
        adapter = Adapter(verbose=True)
        chunks = adapter.get_chunks(texts)

        # Step 4: Initialize the embedder and get embeddings for the documents
        log_placeholder.text("Embedding documents...")
        doc_embedder = Embedder(embedding_model="all-MiniLM-l6-v2", verbose=True)
        embeddings = doc_embedder.embed_documents(chunks)

        # Step 5: Initialize the vector database and insert embeddings
        log_placeholder.text("Inserting embeddings into the vector database...")
        vdb = VectorDatabase(verbose=True)
        success = vdb.insert_embeddings(embeddings, chunks)

        if success:
            log_placeholder.text("Embeddings inserted successfully into the vector database.")
        else:
            log_placeholder.text("Failed to insert embeddings into the vector database.")

        # Step 6: Example usage for query embedding
        log_placeholder.text("Embedding query...")
        llm_query_embedding = doc_embedder.embed_query(llm_query)

        # Step 7: Perform a search in the vector database
        log_placeholder.text("Performing search query in vector database...")
        search_results = vdb.search_query(llm_query_embedding, chunk_count=chunk_count)

        # Ensure search results return the correct number of values
        if len(search_results) != chunk_count:
            raise ValueError("Search query did not return the expected number of values")

        # Extract chunks and similarity scores from search results
        search_chunks = [(chunk, similarity) for chunk, similarity in search_results]

        # Step 8: Display search results
        log_placeholder.empty()
        st.subheader("Search Results as context:")
        for i, (chunk, similarity) in enumerate(search_chunks):
            st.write(f"Result {i+1}:")
            st.write(f"{chunk}")
            st.write(f"Similarity: {similarity}\n")

        # Step 9: Load DistilBART model and tokenizer
        log_placeholder.text("Generating answer using DistilBART...")
        model_name = 'lxyuan/distilbart-finetuned-summarization'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Combine the search results as context
        f_chunks = [chunk for chunk, similarity in search_chunks if similarity>0.5]
        context = " ".join(f_chunks)

        if(len(f_chunks)> 0):
        
            # Generate answer
            prompt = prompt_template.format(context=context, question=llm_query)
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
        else:
            generated_answer = "Insufficient information to answer the question."

        st.subheader(f"Generated Answer:\n")
        st.write(generated_answer)

    except Exception as e:
        st.error(f"An error occurred: {e}")
