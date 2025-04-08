import os
import streamlit as st
import requests
import json
import time
from PyPDF2 import PdfReader
import tempfile
from datetime import datetime
#from dotenv import load_dotenv

# Загрузка переменных окружения
#load_dotenv()

# # Получение API ключа
# api_key = os.getenv("DEEPSEEK_API_KEY")
# if not api_key:
#     st.error("API ключ не найден. Пожалуйста, создайте файл .env с DEEPSEEK_API_KEY")
#     st.stop()

hide_github_icon = """
<style>
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK { 
    display: none !important; 
}
#MainMenu, footer, header { 
    display: none !important; 
}
</style>
"""

st.markdown(hide_github_icon, unsafe_allow_html=True)


st.sidebar.write("[Uzun Demir](https://uzundemir.github.io/)") #[Github](https://github.com/UzunDemir)     [Linkedin](https://www.linkedin.com/in/uzundemir/)     
st.sidebar.write("[Github](https://github.com/UzunDemir)")
st.sidebar.write("[Linkedin](https://www.linkedin.com/in/uzundemir/)")
st.sidebar.title("Описание проекта")
st.sidebar.title("Handwritten Digits Classifier MNIST")
st.sidebar.divider()
st.sidebar.write(
        """
                                       
                     Эта приложка выполнена в рамках практической работы по модулю Computer Vision курса Machine Learning Advanced от Skillbox.
                     
                     1. Вначале была обучена модель распознавания рукописных цифр на базе MNIST (Modified National Institute of Standards and Technology database).
                     Точность на тестовой выборке датасета должна быть не ниже 68%. Я использовал много разных моделей и остановил свой выбор на сверточной нейронной сети (Convolutional Neural Network, CNN)
                     которая показала точность на тестовом наборе данных: 0.99.
                     Ноутбук с исследованиями можно посмотреть [здесь.](https://github.com/UzunDemir/mnist_777/blob/main/RESEARCH%26MODEL/prepare_model.ipynb)
                     2. Вторым шагом необходимо было обернуть готовую модель в сервис и запустить её как часть веб-приложения для распознавания самостоятельно написанных символов. 
                     После этого нужно было создать docker-образ и запустить приложение в docker-контейнере.
                     3. Я решил сделать [полноценное приложение, которое загружает изображение цифры и предсказывает ее](https://mnistpred.streamlit.app/). 
                     Но как злостный перфекционист, я подумал: а что если самому рисовать цифру и пусть модель ее предсказывает! 
                     Немного поискал как реализовать эту идею  и остановил свой выбор на Streamlit.
                     И вот что получилось!
                     
                     """
    )

# Устанавливаем стиль для центрирования элементов
st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        /height: 5vh;
        text-align: center;
        flex-direction: column;
        margin-top: 0vh;  /* отступ сверху */
    }
    .github-icon:hover {
        color: #4078c0; /* Изменение цвета при наведении */
    }
    </style>
    <div class="center">
        <img src="https://github.com/UzunDemir/mnist_777/blob/main/200w.gif?raw=true">
        <h1>AI-ассистент по тестам</h1>
        <p> (строго по учебным материалам)</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
# Настройки для канвы
stroke_width = 10
stroke_color = "black"
bg_color = "white"
drawing_mode = "freedraw"

# Получение API ключа
api_key = st.secrets.get("DEEPSEEK_API_KEY")
if not api_key:
    st.error("API ключ не настроен. Пожалуйста, добавьте его в Secrets.")
    st.stop()

url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

class KnowledgeBase:
    def __init__(self):
        self.documents = {}
        self.uploaded_files = []
    
    def load_pdf(self, file_content, file_name):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                self.documents[file_name] = text
                self.uploaded_files.append(file_name)
                return True
        except Exception as e:
            st.error(f"Ошибка загрузки PDF: {e}")
            return False
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    def get_all_text(self):
        return "\n\n".join([f"### {name}\n{text}" for name, text in self.documents.items()])
    
    def get_document_names(self):
        return self.uploaded_files

# Инициализация
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = KnowledgeBase()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Интерфейс Streamlit
#st.title("AI-ассистент по тестам (строго по учебным материалам)")

# Загрузка документов
uploaded_files = st.file_uploader("Загрузить PDF", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.knowledge_base.get_document_names():
            success = st.session_state.knowledge_base.load_pdf(uploaded_file.getvalue(), uploaded_file.name)
            if success:
                st.success(f"Файл {uploaded_file.name} успешно загружен")

# Отображение загруженных документов
if st.session_state.knowledge_base.get_document_names():
    st.subheader("📚 Загруженные документы:")
    for doc in st.session_state.knowledge_base.get_document_names():
        st.markdown(f"- {doc}")
else:
    st.info("ℹ️ Документы не загружены")

# Отображение истории сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ввод вопроса
if prompt := st.chat_input("Введите ваш вопрос..."):
    # Добавляем вопрос в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Подготавливаем контекст из базы знаний
    context = st.session_state.knowledge_base.get_all_text()
    full_prompt = f""""Answer strictly based on the educational materials provided below.
    Respond in the same language the question is written in.
    If the answer is not found in the materials, reply with: 'Answer not found in the materials'.
    
    Materials:
    {context}
    
    Question: {prompt}"""
    
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": True
    }
    
    # Показываем индикатор выполнения
    with st.spinner("Ищем ответ..."):
        start_time = datetime.now()
        
        try:
            response = requests.post(url, headers=headers, json=data, stream=True)
            
            if response.status_code == 200:
                full_response = ""
                message_placeholder = st.empty()
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        
                        if decoded_line.startswith("data:"):
                            json_data = decoded_line[5:].strip()
                            
                            if json_data == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(json_data)
                                if 'choices' in data and len(data['choices']) > 0:
                                    chunk_content = data['choices'][0]['delta'].get('content', '')
                                    if chunk_content:
                                        full_response += chunk_content
                                        message_placeholder.markdown(f"🤖 {full_response}")
                                        time.sleep(0.05)
                            except json.JSONDecodeError:
                                continue
                
                # Добавляем ответ в историю
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                with st.chat_message("assistant"):
                    if "ответ не найден" in full_response.lower():
                        st.markdown(f"{full_response} ❌")
                    else:
                        st.markdown(f"{full_response} ✅")
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                st.info(f"⏱️ Поиск ответа занял {duration:.2f} секунд")
            else:
                st.error(f"Ошибка API: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")

# Кнопка очистки чата
if st.button("Очистить чат"):
    st.session_state.messages = []
    st.rerun()
