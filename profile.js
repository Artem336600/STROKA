let currentUser = null;

// Проверяем авторизацию при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    const token = localStorage.getItem('token');
    const userStr = localStorage.getItem('user');
    
    if (!token || !userStr) {
        window.location.href = '/';
        return;
    }

    try {
        currentUser = JSON.parse(userStr);
        displayProfile();
    } catch (error) {
        console.error('Error parsing user data:', error);
        window.location.href = '/';
    }
});

// Отображение профиля
function displayProfile() {
    if (!currentUser) return;

    // Отображаем Telegram
    document.getElementById('profile-telegram').textContent = '@' + currentUser.telegram;

    // Отображаем описание
    const aboutElement = document.getElementById('profile-about');
    aboutElement.textContent = currentUser.about || 'Описание не указано';

    // Отображаем теги
    const tagsContainer = document.getElementById('profile-tags');
    tagsContainer.innerHTML = '';
    
    if (currentUser.tags && Array.isArray(currentUser.tags)) {
        currentUser.tags.forEach(tag => {
            const tagElement = document.createElement('span');
            tagElement.className = 'tag';
            tagElement.textContent = tag;
            tagsContainer.appendChild(tagElement);
        });
    } else {
        tagsContainer.innerHTML = '<p class="no-tags">Нет тегов</p>';
    }
}

// Обработчик выхода
document.getElementById('logout-btn').addEventListener('click', () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/';
});

// Обработчики для кнопок чатов и уведомлений
document.getElementById('chats-btn').addEventListener('click', () => {
    window.location.href = '/chats.html';
});

document.getElementById('notifications-btn').addEventListener('click', () => {
    window.location.href = '/notifications.html';
}); 