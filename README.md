# RAG-for-Q-A
## An end-to-end RAG pipeline for question answering from an uploaded PDF

This project is a Streamlit-based application that allows users to upload PDF documents and ask questions about the content. It uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate answers by leveraging embeddings and a vector database. The pipeline includes steps for document reading, chunking, embedding, vector storage, and language model inference.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Use](#how-to-use)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

## Overview

This application reads a PDF document, extracts its text, and processes it into chunks suitable for embedding. The chunks are then embedded using a pre-trained model and stored in a vector database. When a user inputs a query, the application retrieves the most relevant chunks based on the query's embedding and generates an answer using a language model.

## Features

- *PDF Document Processing:* Extract text and paragraphs from PDF files.
- *Text Chunking:* Divide documents into manageable chunks for embedding.
- *Embeddings Generation:* Use a pre-trained Hugging Face model to generate embeddings.
- *Vector Database:* Store and retrieve document chunks using FAISS for efficient similarity search.
- *Language Model Inference:* Generate answers using a pre-trained language model based on retrieved chunks.
- *Streamlit Interface:* Easy-to-use web interface for uploading documents, entering queries, and displaying results.

## Technologies Used

- *Python*: Core programming language.
- *Streamlit*: For building the web application interface.
- *PyPDF2*: To extract text from PDF documents.
- *LangChain*: For text chunking and embeddings management.
- *FAISS*: For efficient similarity search in the vector database.
- *Hugging Face Transformers*: For embedding generation and language model inference.
- *DistilBART*: Pre-trained model used for generating answers.

## Project Structure

```plaintext
.
├── adapter.py         # Handles text chunking and preparation for embedding.
├── embedder.py        # Manages embedding generation using Hugging Face models.
├── reader.py          # Extracts text and paragraphs from PDF documents.
├── vector_db.py       # Manages the FAISS-based vector database for storing and retrieving embeddings.
├── app.py             # Main Streamlit application file.
└── README.md          # Project documentation.


