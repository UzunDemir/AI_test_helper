import os
import streamlit as st
import requests
import json
import time
from PyPDF2 import PdfReader
import tempfile
from datetime import datetime
from transformers import GPT2Tokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# TOKENIZER
# -----------------------------

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# -----------------------------
# STREAMLIT CONFIG
# -----------------------------

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

.center {
    display:flex;
    justify-content:center;
    align-items:center;
    text-align:center;
    flex-direction:column;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# SIDEBAR TOGGLE
# -----------------------------

if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

toggle = st.button("☰", help="Показать / скрыть меню")

if toggle:
    st.session_state.sidebar_open = not st.session_state.sidebar_open


# -----------------------------
# SIDEBAR
# -----------------------------

if st.session_state.sidebar_open:

    with st.sidebar:

        st.title("TEST-passer")
        st.subheader("AI-ассистент по тестам")

        st.divider()

        st.write(
        """
Это приложение помогает студентам отвечать на тесты
**строго по учебным материалам**.

### Возможности

• загрузка PDF  
• создание векторной базы  
• TF-IDF поиск  
• поиск по релевантным фрагментам  
• генерация ответа через DeepSeek  

### Улучшения

• динамический чанкинг  
• поиск 3 лучших фрагментов  
• низкая температура модели  
• источники (документ + страница)

### Планируется

• загрузка вопросов из скриншотов  
• ансамбль моделей  
• ускорение поиска
"""
        )


# -----------------------------
# HEADER
# -----------------------------

st.markdown("""

<div class="center">

<img src="https://github.com/UzunDemir/mnist_777/blob/main/200w.gif?raw=true">

# TEST-passer

### AI-ассистент по тестам

*(строго по учебным материалам)*

</div>

""", unsafe_allow_html=True)

st.divider()


# -----------------------------
# API KEY
# -----------------------------

api_key = st.secrets.get("DEEPSEEK_API_KEY")

if not api_key:
    st.error("API ключ не найден")
    st.stop()

url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


# -----------------------------
# DATA STRUCTURES
# -----------------------------

class DocumentChunk:

    def __init__(self, text, doc_name, page_num):

        self.text = text
        self.doc_name = doc_name
        self.page_num = page_num


class KnowledgeBase:

    def __init__(self):

        self.chunks = []
        self.uploaded_files = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.doc_texts = []


    # -----------------------------
    # TEXT SPLIT
    # -----------------------------

    def split_text(self, text, max_tokens=2000):

        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = ""

        for para in paragraphs:

            para = para.strip()

            if not para:
                continue

            tokens = tokenizer.tokenize(para)

            if len(tokenizer.tokenize(current_chunk + para)) > max_tokens:

                if current_chunk:

                    chunks.append(current_chunk)
                    current_chunk = para

                else:

                    chunks.append(para)
                    current_chunk = ""

            else:

                if current_chunk:

                    current_chunk += "\n\n" + para

                else:

                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


    # -----------------------------
    # LOAD PDF
    # -----------------------------

    def load_pdf(self, file_content, file_name):

        try:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name

            with open(tmp_file_path, "rb") as file:

                reader = PdfReader(file)

                for page_num, page in enumerate(reader.pages):

                    page_text = page.extract_text()

                    if page_text:

                        chunks = self.split_text(page_text)

                        for chunk in chunks:

                            self.chunks.append(
                                DocumentChunk(
                                    text=chunk,
                                    doc_name=file_name,
                                    page_num=page_num + 1,
                                )
                            )

                            self.doc_texts.append(chunk)

                if self.chunks:

                    self.uploaded_files.append(file_name)

                    self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

                    return True

                else:

                    st.error("Не удалось извлечь текст")

                    return False

        except Exception as e:

            st.error(f"Ошибка PDF: {e}")

            return False

        finally:

            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)


    # -----------------------------
    # SEARCH
    # -----------------------------

    def find_most_relevant_chunks(self, query, top_k=3):

        if not self.chunks:
            return []

        query_vec = self.vectorizer.transform([query])

        similarities = cosine_similarity(query_vec, self.tfidf_matrix)

        top_indices = np.argsort(similarities[0])[-top_k:][::-1]

        return [
            (
                self.chunks[i].text,
                self.chunks[i].doc_name,
                self.chunks[i].page_num,
            )
            for i in top_indices
            if similarities[0][i] > 0.1
        ]


    def get_document_names(self):

        return self.uploaded_files


# -----------------------------
# SESSION STATE
# -----------------------------

if "knowledge_base" not in st.session_state:

    st.session_state.knowledge_base = KnowledgeBase()

if "messages" not in st.session_state:

    st.session_state.messages = []


# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_files = st.file_uploader(
    "Загрузить учебные материалы (PDF)",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        if (
            uploaded_file.name
            not in st.session_state.knowledge_base.get_document_names()
        ):

            success = st.session_state.knowledge_base.load_pdf(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )

            if success:

                st.success(f"{uploaded_file.name} загружен")


# -----------------------------
# SHOW DOCUMENTS
# -----------------------------

if st.session_state.knowledge_base.get_document_names():

    st.subheader("📚 Загруженные документы")

    for doc in st.session_state.knowledge_base.get_document_names():

        st.write(doc)

else:

    st.info("Документы пока не загружены")


# -----------------------------
# CHAT HISTORY
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# -----------------------------
# USER INPUT
# -----------------------------

if prompt := st.chat_input("Введите вопрос"):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    relevant_chunks = st.session_state.knowledge_base.find_most_relevant_chunks(
        prompt
    )

    if not relevant_chunks:

        response_text = "Ответ не найден в материалах ❌"

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

        with st.chat_message("assistant"):

            st.markdown(response_text)

    else:

        context = "\n\n".join(

            [
                f"Документ: {doc}, страница {page}\n{text}"
                for text, doc, page in relevant_chunks
            ]

        )

        full_prompt = f"""
Answer strictly using the educational materials.

Question:
{prompt}

Relevant materials:
{context}
"""

        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 2000,
            "temperature": 0.1,
        }

        with st.spinner("Поиск ответа..."):

            start_time = datetime.now()

            response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:

                answer = response.json()["choices"][0]["message"]["content"]

                sources = "\n\nИсточники:\n" + "\n".join(
                    [f"{doc}, стр {page}" for _, doc, page in relevant_chunks]
                )

                answer += sources

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

                with st.chat_message("assistant"):

                    st.markdown(answer + " ✅")

                duration = (datetime.now() - start_time).total_seconds()

                st.info(f"⏱️ {duration:.2f} сек")

            else:

                st.error(response.text)


# -----------------------------
# CLEAR CHAT
# -----------------------------

if st.button("Очистить чат"):

    st.session_state.messages = []

    st.rerun()
