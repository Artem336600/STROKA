import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import time

# Инициализация клиента Mistral
client = MistralClient(api_key="InDPitkUkV2JX5S1wdlWZwIfee6wTwLc")

# Настройка страницы
st.set_page_config(
    page_title="Mistral AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стилизация
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F5F5F5;
    }
    </style>
""", unsafe_allow_html=True)

# Заголовок
st.title("💬 Чат с Mistral AI")

# Инициализация истории чата в session_state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Отображение истории сообщений
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <b>Вы:</b> {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <b>Mistral AI:</b> {message["content"]}
                </div>
            """, unsafe_allow_html=True)

# Поле ввода
user_input = st.text_input("Введите ваш запрос:", key="user_input")

# Кнопка отправки
if st.button("Отправить"):
    if user_input:
        # Добавляем сообщение пользователя в историю
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Создаем сообщение для отправки
        messages = [
            ChatMessage(role="user", content=user_input)
        ]
        
        # Получаем ответ от модели
        with st.spinner("Mistral AI думает..."):
            chat_response = client.chat(
                model="mistral-tiny",
                messages=messages
            )
            
            # Добавляем ответ ассистента в историю
            st.session_state.messages.append({
                "role": "assistant",
                "content": chat_response.choices[0].message.content
            })
            
            # Обновляем страницу для отображения нового сообщения
            st.experimental_rerun()
    else:
        st.warning("Пожалуйста, введите запрос")

# Кнопка очистки истории
if st.sidebar.button("Очистить историю"):
    st.session_state.messages = []
    st.experimental_rerun()

# Информация в сайдбаре
st.sidebar.markdown("""
    ### О приложении
    Это приложение использует Mistral AI для генерации ответов на ваши запросы.
    
    ### Инструкция
    1. Введите ваш запрос в текстовое поле
    2. Нажмите кнопку "Отправить"
    3. Дождитесь ответа от Mistral AI
    
    ### Примечание
    Используется модель mistral-tiny для быстрых ответов.
""") 