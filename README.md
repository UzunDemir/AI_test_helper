

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


1. Upload PDF → 2. Split into chunks → 3. Generate embeddings → 4. Store in FAISS
                           ↓
5. Ask question → 6. HyDE expansion → 7. Hybrid search → 8. Rerank → 9. LLM answer
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/test-passer.git
cd test-passer

# Install dependencies
pip install -r requirements.txt

# Set up API key in Streamlit secrets
echo "DEEPSEEK_API_KEY='your-key-here'" > .streamlit/secrets.toml

# Run the app
streamlit run app.py
```

## 🔑 Environment Variables

Create `.streamlit/secrets.toml`:

```toml
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

## 🎮 Usage

1. Launch the app
2. Upload one or more PDF files
3. Ask questions in natural language
4. Receive answers with source citations

## 📊 Example

**User:** *What is the capital of France?*

**Assistant:** According to *Geography.pdf* (page 12), Paris is the capital of France. It has been the political and cultural center since the 12th century.

## 🧩 Project Structure

```
test-passer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── secrets.toml    # API keys (gitignored)
└── README.md           # This file
```

## 🤝 Contributing

PRs and suggestions are welcome! Feel free to open an issue.

## 📄 License

MIT License — free to use, modify, and distribute.

---

⭐ **Star this repo** if you find it useful!
```

This README:
- Uses emojis and badges for visual appeal
- Clearly explains the value proposition
- Shows the tech stack at a glance
- Includes a simple architecture diagram
- Provides setup and usage instructions
- Keeps it short but informative
