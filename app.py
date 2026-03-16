import os
import streamlit as st
import requests
import tempfile
import numpy as np
import faiss
import time

from datetime import datetime
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------- STREAMLIT CONFIG ----------------

st.set_page_config(layout="wide", initial_sidebar_state="auto")

# ---------------- MODEL CACHE ----------------

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embedder = load_embedder()
reranker = load_reranker()

# ---------------- DEEPSEEK API ----------------

api_key = st.secrets.get("DEEPSEEK_API_KEY")

url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# ---------------- DATA STRUCTURES ----------------

class DocumentChunk:

    def __init__(self, text, doc_name, page_num):
        self.text = text
        self.doc_name = doc_name
        self.page_num = page_num


# ---------------- KNOWLEDGE BASE ----------------

class KnowledgeBase:

    def __init__(self):

        self.chunks = []
        self.embeddings = []

        self.index = None

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.doc_texts = []

        self.uploaded_files = []

    # ---------- CHUNKING ----------

    def split_text(self, text, max_chars=1500):

        paragraphs = text.split("\n\n")
        chunks = []

        current = ""

        for para in paragraphs:

            if len(current) + len(para) < max_chars:
                current += para + "\n\n"

            else:
                chunks.append(current)
                current = para

        if current:
            chunks.append(current)

        return chunks

    # ---------- PDF LOADING ----------

    def load_pdf(self, file_content, file_name):

        tmp_path = None

        try:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name

            reader = PdfReader(tmp_path)

            for page_num, page in enumerate(reader.pages):

                text = page.extract_text()

                if text:

                    chunks = self.split_text(text)

                    for chunk in chunks:

                        c = DocumentChunk(chunk, file_name, page_num + 1)

                        self.chunks.append(c)

                        self.doc_texts.append(chunk)

                        emb = embedder.encode(chunk)

                        self.embeddings.append(emb)

            self.uploaded_files.append(file_name)

            # TF-IDF

            self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

            # FAISS

            vectors = np.array(self.embeddings).astype("float32")

            dim = vectors.shape[1]

            self.index = faiss.IndexFlatL2(dim)
            self.index.add(vectors)

            return True

        except Exception as e:

            st.error(f"PDF error: {e}")
            return False

        finally:

            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ---------- HYDE ----------

    def generate_hypothetical(self, query):

        prompt = f"""
Write a short paragraph that answers the question.

Question:
{query}

Answer:
"""

        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.3
        }

        try:

            r = requests.post(url, headers=headers, json=data, timeout=30)

            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']

        except:
            pass

        return query

    # ---------- SEMANTIC SEARCH ----------

    def semantic_search(self, query, top_k=6):

        q = embedder.encode([query]).astype("float32")

        distances, indices = self.index.search(q, top_k)

        results = []

        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results

    # ---------- KEYWORD SEARCH ----------

    def keyword_search(self, query, top_k=6):

        q = self.vectorizer.transform([query])

        sims = cosine_similarity(q, self.tfidf_matrix)

        idx = np.argsort(sims[0])[-top_k:]

        results = []

        for i in idx:
            results.append(self.chunks[i])

        return results

    # ---------- HYBRID + RERANK ----------

    def retrieve(self, query, top_k=3):

        hypothetical = self.generate_hypothetical(query)

        search_query = query + " " + hypothetical

        semantic = self.semantic_search(search_query)

        keyword = self.keyword_search(search_query)

        combined = semantic + keyword

        unique = list({c.text: c for c in combined}.values())

        pairs = [[query, c.text] for c in unique]

        scores = reranker.predict(pairs)

        ranked = sorted(
            zip(unique, scores),
            key=lambda x: x[1],
            reverse=True
        )

        top = ranked[:top_k]

        return [x[0] for x in top]


# ---------------- SESSION ----------------

if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()

if "messages" not in st.session_state:
    st.session_state.messages = []

kb = st.session_state.kb

# ---------------- UI ----------------

st.title("TEST-passer")
st.write("AI ассистент для тестов (RAG)")

# ---------- FILE UPLOAD ----------

uploaded_files = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    for f in uploaded_files:

        if f.name not in kb.uploaded_files:

            if kb.load_pdf(f.read(), f.name):

                st.success(f"{f.name} loaded")


# ---------- CHAT HISTORY ----------

for m in st.session_state.messages:

    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# ---------- USER INPUT ----------

if prompt := st.chat_input("Ask question"):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    start = datetime.now()

    chunks = kb.retrieve(prompt)

    if not chunks:

        answer = "Answer not found in materials."

    else:

        context = ""

        for c in chunks:

            context += f"""
Document: {c.doc_name}
Page: {c.page_num}

{c.text}

"""

        system_prompt = f"""
You are an AI assistant answering exam questions.

Rules:

1. Use ONLY the provided materials.
2. If answer not found say:
"Answer not found in materials."

Question:
{prompt}

Materials:
{context}
"""

        data = {

            "model": "deepseek-chat",

            "messages": [
                {"role": "user", "content": system_prompt}
            ],

            "temperature": 0.1,
            "max_tokens": 1000
        }

        r = requests.post(url, headers=headers, json=data, timeout=60)

        answer = r.json()['choices'][0]['message']['content']

        sources = "\n\nSources:\n"

        for c in chunks:
            sources += f"- {c.doc_name} page {c.page_num}\n"

        answer += sources

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

    end = datetime.now()

    st.info(
        f"Search time {(end-start).total_seconds():.2f} sec"
    )

# ---------- CLEAR CHAT ----------

if st.button("Clear chat"):

    st.session_state.messages = []
    st.rerun()
