let currentUser = null;

// Проверяем авторизацию при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    const token = localStorage.getItem('token');
    const userStr = localStorage.getItem('user');
    
    if (token && userStr) {
        currentUser = JSON.parse(userStr);
        loadNotifications();
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

// Загрузка уведомлений
async function loadNotifications() {
    try {
        const response = await fetch('http://localhost:5000/api/notifications', {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        if (response.ok) {
            const notifications = await response.json();
            displayNotifications(notifications);
        } else {
            console.error('Ошибка при загрузке уведомлений');
        }
    } catch (error) {
        console.error('Ошибка при загрузке уведомлений:', error);
    }
}

// Отображение уведомлений
function displayNotifications(notifications) {
    const notificationsList = document.querySelector('.notifications-list');
    notificationsList.innerHTML = '';

    if (notifications.length === 0) {
        notificationsList.innerHTML = '<p class="no-data">У вас пока нет уведомлений</p>';
        return;
    }

    notifications.forEach(notification => {
        const notificationElement = document.createElement('div');
        notificationElement.className = `notification-item ${notification.read ? 'read' : 'unread'}`;
        notificationElement.innerHTML = `
            <div class="notification-content">
                <p>${notification.message}</p>
                <span class="notification-time">${formatTime(notification.created_at)}</span>
            </div>
            ${!notification.read ? '<span class="unread-dot"></span>' : ''}
        `;
        notificationElement.addEventListener('click', () => {
            markNotificationAsRead(notification.id);
        });
        notificationsList.appendChild(notificationElement);
    });
}

// Отметить уведомление как прочитанное
async function markNotificationAsRead(notificationId) {
    try {
        const response = await fetch(`http://localhost:5000/api/notifications/${notificationId}/read`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        if (response.ok) {
            loadNotifications(); // Перезагружаем уведомления
        } else {
            console.error('Ошибка при отметке уведомления как прочитанного');
        }
    } catch (error) {
        console.error('Ошибка при отметке уведомления как прочитанного:', error);
    }
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