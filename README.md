# 📝 TEST-passer

### AI-Powered Test Assistant

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-vector-blue)](https://github.com/facebookresearch/faiss)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-API-green)](https://deepseek.com)

## ✨ Overview

**TEST-passer** is an intelligent RAG (Retrieval-Augmented Generation) system that answers questions strictly based on uploaded educational materials. Perfect for students, educators, and professionals preparing for tests and exams.

## 🚀 Key Features

- **📄 PDF Upload** – Upload any educational PDF materials
- **🔍 Hybrid Search** – Combines semantic (FAISS) + keyword (TF-IDF) retrieval
- **🎯 HyDE Query Expansion** – Improves search relevance using AI-generated context
- **⚡ Cross-Encoder Reranking** – Prioritizes the most relevant chunks
- **💬 Strict Context** – Answers only from uploaded materials, no hallucinations
- **📚 Source Citations** – Shows document name and page numbers for each answer

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **Reranking** | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| **LLM** | DeepSeek Chat API |
| **Frontend** | Streamlit |
| **PDF Processing** | PyPDF2 |

## 🧠 How It Works
