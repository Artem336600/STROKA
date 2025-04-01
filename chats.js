let currentUser = null;

// Базовый URL API
const API_BASE_URL = '';

// Проверяем авторизацию при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    const token = localStorage.getItem('token');
    const userStr = localStorage.getItem('user');
    
    if (token && userStr) {
        currentUser = JSON.parse(userStr);
        loadChats();
    } else {
        window.location.href = '/';
    }
});

// Обработчик кнопки "Назад"
document.getElementById('back-btn').addEventListener('click', () => {
    window.location.href = '/';
});

// Обработчик кнопки профиля
document.getElementById('profile-btn').addEventListener('click', () => {
    window.location.href = '/profile.html';
});

// Загрузка чатов
async function loadChats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/chats`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        if (response.ok) {
            const chats = await response.json();
            displayChats(chats);
        } else {
            console.error('Ошибка при загрузке чатов');
        }
    } catch (error) {
        console.error('Ошибка при загрузке чатов:', error);
    }
}

// Отображение чатов
function displayChats(chats) {
    const chatsList = document.querySelector('.chats-list');
    chatsList.innerHTML = '';

    if (chats.length === 0) {
        chatsList.innerHTML = '<p class="no-data">У вас пока нет чатов</p>';
        return;
    }

    chats.forEach(chat => {
        const chatElement = document.createElement('div');
        chatElement.className = 'chat-item';
        chatElement.innerHTML = `
            <div class="chat-info">
                <h3>${chat.telegram}</h3>
                <p>${chat.lastMessage || 'Нет сообщений'}</p>
            </div>
            <div class="chat-meta">
                <span class="chat-time">${formatTime(chat.lastMessageTime)}</span>
                ${chat.unreadCount > 0 ? `<span class="unread-badge">${chat.unreadCount}</span>` : ''}
            </div>
        `;
        chatElement.addEventListener('click', () => {
            window.location.href = `/chat.html?id=${chat.id}`;
        });
        chatsList.appendChild(chatElement);
    });
}

// Форматирование времени
function formatTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleString('ru-RU', {
        hour: '2-digit',
        minute: '2-digit'
    });
} 