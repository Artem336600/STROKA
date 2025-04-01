// В начале файла добавим определение базового URL API
const API_BASE_URL = ''; // Пустая строка для относительных путей

// Глобальные переменные
let selectedTags = new Set();
let aiTags = new Set();
let debounceTimer;
let currentUser = null;
let verificationCode = null;
let verificationInterval = null;

let registrationData = {
    telegram: '',
    password: '',
    about: '',
    tags: []
};

// Элементы DOM
const profileBtn = document.getElementById('profile-btn');
const profileDropdown = document.getElementById('profile-dropdown');
const authButtons = document.getElementById('auth-buttons');
const userInfo = document.getElementById('user-info');
const profileTelegram = document.getElementById('profile-telegram');
const loginBtn = document.getElementById('login-btn');
const registerBtn = document.getElementById('register-btn');
const logoutBtn = document.getElementById('logout-btn');
const loginModal = document.getElementById('login-modal');
const registerModal = document.getElementById('register-modal');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const searchInput = document.getElementById('search-input');
const searchButton = document.getElementById('search-button');
const selectedTagsContainer = document.getElementById('selected-tags');
const aiTagsContainer = document.getElementById('ai-tags');
const addTagsBtn = document.getElementById('add-tags-btn');
const tagsModal = document.getElementById('tags-modal');
const tagsGrid = document.getElementById('tags-grid');
const modalTagsGrid = document.getElementById('modal-tags-grid');

// Получаем элементы модальных окон
const chatsModal = document.getElementById('chats-modal');
const notificationsModal = document.getElementById('notifications-modal');

// Получаем кнопки
const chatsButton = document.getElementById('chats-button');
const notificationsButton = document.getElementById('notifications-button');

// Обработчики модальных окон
function showModal(modal) {
    modal.style.display = 'block';
}

function hideModal(modal) {
    modal.style.display = 'none';
}

// Закрытие модальных окон при клике вне их области
window.onclick = function(event) {
    if (event.target === loginModal) hideModal(loginModal);
    if (event.target === registerModal) hideModal(registerModal);
    if (event.target === tagsModal) hideModal(tagsModal);
    if (event.target === chatsModal) hideModal(chatsModal);
    if (event.target === notificationsModal) hideModal(notificationsModal);
}

// Обработчики форм
loginForm.addEventListener('submit', login);

async function login(event) {
    event.preventDefault();
    const telegram = document.getElementById('login-telegram').value;
    const password = document.getElementById('login-password').value;

    try {
        console.log('Attempting login with:', { telegram }); // Логируем попытку входа

        const response = await fetch(`${API_BASE_URL}/api/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ telegram, password })
        });

        console.log('Response status:', response.status); // Логируем статус ответа
        const data = await response.json();
        console.log('Login response:', data); // Логируем ответ сервера
        
        if (response.ok) {
            console.log('Login successful, setting user data'); // Логируем успешный вход
            localStorage.setItem('token', data.token);
            localStorage.setItem('user', JSON.stringify(data.user));
            currentUser = data.user;
            
            // Скрываем модальное окно входа
            hideModal(loginModal);
            
            // Перенаправляем на страницу профиля
            window.location.href = '/profile.html';
        } else {
            console.error('Login error:', data.error); // Логируем ошибку
            showNotification(data.error || 'Ошибка при входе', 'error');
        }
    } catch (error) {
        console.error('Login error:', error); // Логируем ошибку
        showNotification('Ошибка при входе', 'error');
    }
}

registerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    registrationData.telegram = document.getElementById('register-telegram').value;
    registrationData.password = document.getElementById('register-password').value;

    try {
        // Проверяем, существует ли пользователь
        const checkResponse = await fetch(`${API_BASE_URL}/api/check-user`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ telegram: registrationData.telegram })
        });

        const checkData = await checkResponse.json();
        
        if (checkResponse.ok) {
            // Если пользователь не существует, запрашиваем код подтверждения
            const verificationResponse = await fetch(`${API_BASE_URL}/api/request-verification`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ telegram: registrationData.telegram })
            });

            const verificationData = await verificationResponse.json();
            
            if (verificationResponse.ok) {
                // Показываем второй шаг с кодом подтверждения
                document.getElementById('register-step-1').style.display = 'none';
                document.getElementById('register-step-2').style.display = 'block';
                
                // Отображаем код подтверждения
                verificationCode = verificationData.code;
                document.getElementById('verification-code-display').textContent = verificationCode;
                
                // Добавляем обработчик для копирования кода
                document.getElementById('copy-code-btn').addEventListener('click', () => {
                    navigator.clipboard.writeText(verificationCode);
                    showNotification('Код скопирован', 'success');
                });

                // Начинаем проверку подтверждения
                startVerificationCheck();
            } else {
                showNotification(verificationData.error, 'error');
            }
        } else {
            showNotification(checkData.error, 'error');
        }
    } catch (error) {
        console.error('Error during registration:', error);
        showNotification('Ошибка при регистрации', 'error');
    }
});

function startVerificationCheck() {
    const statusElement = document.getElementById('verification-status');
    const checkButton = document.getElementById('check-verification-btn');
    
    // Очищаем предыдущий интервал, если он существует
    if (verificationInterval) {
        clearInterval(verificationInterval);
    }

    checkButton.addEventListener('click', async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/check-verification`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    telegram: registrationData.telegram,
                    code: verificationCode
                })
            });

            const data = await response.json();
            
            if (response.ok && data.verified) {
                // Подтверждение успешно
                statusElement.textContent = 'Подтверждено!';
                statusElement.className = 'verified';
                showNotification('Telegram подтвержден!', 'success');
                
                // Переходим к следующему шагу
                document.getElementById('register-step-2').style.display = 'none';
                document.getElementById('register-step-3').style.display = 'block';
                
                // Очищаем интервал проверки
                if (verificationInterval) {
                    clearInterval(verificationInterval);
                }
            } else {
                showNotification('Подтверждение не получено', 'error');
            }
        } catch (error) {
            console.error('Error checking verification:', error);
            showNotification('Ошибка при проверке подтверждения', 'error');
        }
    });

    // Автоматическая проверка каждые 5 секунд
    verificationInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/check-verification`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    telegram: registrationData.telegram,
                    code: verificationCode
                })
            });

            const data = await response.json();
            
            if (response.ok && data.verified) {
                statusElement.textContent = 'Подтверждено!';
                statusElement.className = 'verified';
                showNotification('Telegram подтвержден!', 'success');
                
                // Переходим к следующему шагу
                document.getElementById('register-step-2').style.display = 'none';
                document.getElementById('register-step-3').style.display = 'block';
                
                // Очищаем интервал
                clearInterval(verificationInterval);
            }
        } catch (error) {
            console.error('Error checking verification:', error);
        }
    }, 5000);
}

// Обработчик поля about
document.getElementById('register-about').addEventListener('input', async (e) => {
    const about = e.target.value;
    if (about.length > 10) { // Начинаем поиск тегов только если текст достаточно длинный
        try {
            const response = await fetch(`${API_BASE_URL}/api/suggest-tags`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: about })
            });

            const data = await response.json();
            if (response.ok) {
                const suggestedTags = document.querySelector('.suggested-tags');
                suggestedTags.innerHTML = data.tags.map(tag => `
                    <div class="tag-item" data-tag="${tag}">${tag}</div>
                `).join('');

                // Добавляем обработчики для тегов
                suggestedTags.querySelectorAll('.tag-item').forEach(item => {
                    item.addEventListener('click', () => {
                        const tag = item.dataset.tag;
                        if (registrationData.tags.includes(tag)) {
                            registrationData.tags = registrationData.tags.filter(t => t !== tag);
                            item.classList.remove('selected');
                        } else {
                            registrationData.tags.push(tag);
                            item.classList.add('selected');
                        }
                    });
                });
            }
        } catch (error) {
            console.error('Error fetching suggestions:', error);
        }
    }
});

// Обработчик формы about
document.getElementById('register-about-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    registrationData.about = document.getElementById('register-about').value;

    try {
        const response = await fetch(`${API_BASE_URL}/api/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(registrationData)
        });

        const data = await response.json();
        if (response.ok) {
            currentUser = data.user;
            localStorage.setItem('token', data.token);
            localStorage.setItem('user', JSON.stringify(data.user));
            hideModal(registerModal);
            updateAuthUI();
            showNotification('Регистрация успешна', 'success');
            // Сбрасываем данные регистрации
            registrationData = {
                telegram: '',
                password: '',
                about: '',
                tags: []
            };
            document.getElementById('register-step-1').style.display = 'block';
            document.getElementById('register-step-2').style.display = 'none';
        } else {
            showNotification(data.error, 'error');
        }
    } catch (error) {
        showNotification('Ошибка при регистрации', 'error');
    }
});

document.addEventListener('DOMContentLoaded', () => {
    const savedToken = localStorage.getItem('token');
    const savedUser = localStorage.getItem('user');
    
    if (savedToken && savedUser) {
        try {
            currentUser = JSON.parse(savedUser);
            console.log('Loaded saved user:', currentUser);
            updateAuthUI();
        } catch (error) {
            console.error('Error parsing saved user:', error);
            localStorage.removeItem('token');
            localStorage.removeItem('user');
        }
    }

    // Обработчик клика по кнопке профиля
    if (profileBtn && profileDropdown) {
        profileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            console.log('Profile button clicked');
            
            profileDropdown.classList.toggle('active');
            updateDropdownContent();
        });

        // Закрытие выпадающего меню при клике вне его
        document.addEventListener('click', (e) => {
            if (!profileBtn.contains(e.target) && !profileDropdown.contains(e.target)) {
                profileDropdown.classList.remove('active');
            }
        });
    }

    // Обработчики кнопок авторизации
    if (loginBtn) {
        loginBtn.addEventListener('click', () => {
            profileDropdown.classList.remove('active');
            showModal(loginModal);
        });
    }

    if (registerBtn) {
        registerBtn.addEventListener('click', () => {
            profileDropdown.classList.remove('active');
            showModal(registerModal);
        });
    }
});

function updateDropdownContent() {
    const authButtons = document.getElementById('auth-buttons');
    const userInfo = document.getElementById('user-info');
    const profileDropdown = document.getElementById('profile-dropdown');

    console.log('Updating dropdown content, currentUser:', currentUser);

    if (currentUser) {
        authButtons.style.display = 'none';
        userInfo.style.display = 'block';
        
        userInfo.innerHTML = `
            <div class="profile-header">
                <span class="profile-icon">👤</span>
                <span class="user-telegram">@${currentUser.telegram}</span>
            </div>
            <div class="profile-details">
                <div class="user-about">${currentUser.about || 'Нет описания'}</div>
                <div class="user-tags">
                    ${currentUser.tags && Array.isArray(currentUser.tags) ? 
                        currentUser.tags.map(tag => `<span class="tag">${tag}</span>`).join('') : 
                        '<span class="no-tags">Нет тегов</span>'}
                </div>
            </div>
            <button id="logout-btn" class="auth-button">Выйти</button>
        `;

        // Обработчик для кнопки выхода
        const logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => {
                currentUser = null;
                localStorage.removeItem('token');
                localStorage.removeItem('user');
                updateAuthUI();
                profileDropdown.classList.remove('active');
                showNotification('Вы вышли из системы', 'success');
            });
        }
    } else {
        authButtons.style.display = 'flex';
        userInfo.style.display = 'none';
    }
}

// Обновление UI в зависимости от состояния авторизации
function updateAuthUI() {
    const profileDropdown = document.getElementById('profile-dropdown');
    const authButtons = document.getElementById('auth-buttons');
    const userInfo = document.getElementById('user-info');

    console.log('Updating UI with user:', currentUser); // Отладочный вывод

    if (currentUser) {
        // Показываем информацию о пользователе
        if (authButtons) authButtons.style.display = 'none';
        if (userInfo) {
            userInfo.style.display = 'block';
            
            // Обновляем содержимое информации о пользователе
            userInfo.innerHTML = `
                <div class="profile-header">
                    <span class="profile-icon">👤</span>
                    <span class="user-telegram">@${currentUser.telegram}</span>
                </div>
                <div class="profile-details">
                    <div class="user-about">${currentUser.about || 'Нет описания'}</div>
                    <div class="user-tags">
                        ${currentUser.tags && Array.isArray(currentUser.tags) ? 
                            currentUser.tags.map(tag => `<span class="tag">${tag}</span>`).join('') : 
                            '<span class="no-tags">Нет тегов</span>'}
                    </div>
                </div>
                <button id="logout-btn" class="auth-button">Выйти</button>
            `;

            // Обновляем обработчик для кнопки выхода
            const logoutBtn = document.getElementById('logout-btn');
            if (logoutBtn) {
                logoutBtn.addEventListener('click', () => {
                    currentUser = null;
                    localStorage.removeItem('token');
                    localStorage.removeItem('user');
                    updateAuthUI();
                    showNotification('Вы вышли из системы', 'success');
                    profileDropdown.classList.remove('active');
                });
            }
        }
    } else {
        // Показываем кнопки авторизации
        if (authButtons) authButtons.style.display = 'flex';
        if (userInfo) userInfo.style.display = 'none';
    }
}

// Обработчик кнопки поиска
searchButton.addEventListener('click', async () => {
    const query = searchInput.value.trim();
    console.log('Search button clicked, query:', query);
    console.log('Selected tags:', Array.from(selectedTags));

    if (query) {
        try {
            // Получаем теги из запроса
            const response = await fetch(`${API_BASE_URL}/api/suggest-tags`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });

            const data = await response.json();
            console.log('Suggest tags response:', data);

            if (response.ok) {
                // Добавляем новые теги
                data.tags.forEach(tag => {
                    if (!selectedTags.has(tag)) {
                        addTag(tag);
                    }
                });
                addTagsBtn.style.display = 'block';

                // Ищем пользователей по тегам
                const searchResponse = await fetch(`${API_BASE_URL}/api/search-users`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        tags: Array.from(selectedTags)
                    })
                });

                const searchData = await searchResponse.json();
                console.log('Search users response:', searchData);
                if (searchResponse.ok) {
                    displayResults(searchData.users);
                } else {
                    console.error('Search failed:', searchData.error);
                    showNotification('Ошибка при поиске пользователей', 'error');
                }
            }
        } catch (error) {
            console.error('Error fetching suggestions:', error);
            showNotification('Ошибка при поиске тегов', 'error');
        }
    } else if (selectedTags.size > 0) {
        // Если запрос пустой, но есть выбранные теги, ищем по ним
        try {
            console.log('Searching by selected tags:', Array.from(selectedTags));
            const searchResponse = await fetch(`${API_BASE_URL}/api/search-users`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    tags: Array.from(selectedTags)
                })
            });

            const searchData = await searchResponse.json();
            console.log('Search users response:', searchData);
            if (searchResponse.ok) {
                displayResults(searchData.users);
            } else {
                console.error('Search failed:', searchData.error);
                showNotification('Ошибка при поиске пользователей', 'error');
            }
        } catch (error) {
            console.error('Error searching users:', error);
            showNotification('Ошибка при поиске пользователей', 'error');
        }
    } else {
        showNotification('Введите запрос или выберите теги', 'error');
    }
});

// Обработчик клавиши Enter в поле поиска
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        searchButton.click();
    }
});

// Добавление тега
function addTag(tag) {
    if (!selectedTags.has(tag)) {
        selectedTags.add(tag);
        const tagElement = document.createElement('div');
        tagElement.className = 'tag';
        tagElement.innerHTML = `
            ${tag}
            <span class="remove-tag" data-tag="${tag}">&times;</span>
        `;
        selectedTagsContainer.appendChild(tagElement);
        addTagsBtn.style.display = 'block';
    }
}

// Удаление тега
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('remove-tag')) {
        const tag = e.target.dataset.tag;
        const tagElement = e.target.parentElement;
        selectedTags.delete(tag);
        selectedTagsContainer.removeChild(tagElement);
        
        // Обновляем состояние в модальном окне
        const modalTagItem = modalTagsGrid.querySelector(`.tag-item[data-tag="${tag}"]`);
        if (modalTagItem) {
            modalTagItem.classList.remove('selected');
        }
        
        if (selectedTags.size === 0) {
            addTagsBtn.style.display = 'none';
        }
    }
});

// Функция для отображения результатов поиска
function displayResults(users) {
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = '';

    if (!users || users.length === 0) {
        resultsContainer.innerHTML = '<p class="no-results">Ничего не найдено</p>';
        return;
    }

    users.forEach(result => {
        const userCard = document.createElement('div');
        userCard.className = 'user-card';
        
        // Проверяем, является ли это текущим пользователем
        const isCurrentUser = result.telegram === currentUser?.telegram;
        
        userCard.innerHTML = `
            <div class="user-info">
                <div class="user-telegram">@${result.telegram}</div>
                <div class="user-about">${result.about || 'Нет описания'}</div>
                <div class="user-tags">
                    ${result.tags && Array.isArray(result.tags) ? 
                        result.tags.map(tag => `<span class="tag">${tag}</span>`).join('') : 
                        '<span class="no-tags">Нет тегов</span>'}
                </div>
                ${!isCurrentUser ? '<button class="request-button">Отправить заявку</button>' : ''}
            </div>
        `;

        // Добавляем обработчик для кнопки отправки заявки
        if (!isCurrentUser) {
            const requestButton = userCard.querySelector('.request-button');
            requestButton.addEventListener('click', () => {
                sendRequest(result.telegram);
            });
        }

        resultsContainer.appendChild(userCard);
    });
}

// Функция для отправки заявки
async function sendRequest(targetTelegram) {
    try {
        console.log('Sending request to:', targetTelegram);
        const response = await fetch(`${API_BASE_URL}/api/send-request`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
                target_telegram: targetTelegram
            })
        });

        const data = await response.json();
        console.log('Response:', data);
        
        if (response.ok) {
            showNotification('Заявка отправлена успешно!', 'success');
        } else {
            showNotification(data.error || 'Ошибка при отправке заявки', 'error');
        }
    } catch (error) {
        console.error('Error sending request:', error);
        showNotification('Ошибка при отправке заявки', 'error');
    }
}

// Уведомления
function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Обработчик кнопки "Добавить теги"
addTagsBtn.addEventListener('click', () => {
    showModal(tagsModal);
    updateTagsModal();
});

// Функция обновления модального окна с тегами
function updateTagsModal() {
    modalTagsGrid.innerHTML = '';
    
    // Добавляем все доступные теги
    const allTags = ['JavaScript', 'Python', 'Java', 'C++', 'React', 'Node.js', 'Vue.js', 'Angular', 
                     'Machine Learning', 'Data Science', 'Web Development', 'Mobile Development',
                     'DevOps', 'Cloud Computing', 'Artificial Intelligence', 'Blockchain',
                     'Cybersecurity', 'UI/UX Design', 'Game Development', 'Database'];
    
    allTags.forEach(tag => {
        const tagItem = document.createElement('div');
        tagItem.className = 'tag-item';
        if (selectedTags.has(tag)) {
            tagItem.classList.add('selected');
        }
        tagItem.dataset.tag = tag;
        tagItem.textContent = tag;
        
        tagItem.addEventListener('click', () => {
            if (selectedTags.has(tag)) {
                selectedTags.delete(tag);
                tagItem.classList.remove('selected');
                // Удаляем тег из контейнера выбранных тегов
                const tagElement = selectedTagsContainer.querySelector(`[data-tag="${tag}"]`).parentElement;
                if (tagElement) {
                    selectedTagsContainer.removeChild(tagElement);
                }
            } else {
                selectedTags.add(tag);
                tagItem.classList.add('selected');
                addTag(tag);
            }
            
            if (selectedTags.size === 0) {
                addTagsBtn.style.display = 'none';
            } else {
                addTagsBtn.style.display = 'block';
            }
        });
        
        modalTagsGrid.appendChild(tagItem);
    });
}

// Закрытие модального окна с тегами
const closeTagsModal = document.querySelector('#tags-modal .close');
if (closeTagsModal) {
    closeTagsModal.addEventListener('click', () => {
        hideModal(tagsModal);
    });
}

// Функция для проверки уведомлений
async function checkNotifications() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/notifications`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            const notificationBadge = document.querySelector('.notification-badge');
            
            // Показываем бейдж, если есть уведомления
            if (data.notifications && data.notifications.length > 0) {
                notificationBadge.textContent = data.notifications.length;
                notificationBadge.classList.add('visible');
            } else {
                notificationBadge.classList.remove('visible');
            }

            // Если открыто окно уведомлений, обновляем его содержимое
            const notificationsContainer = document.getElementById('notifications-list');
            if (notificationsContainer && notificationsContainer.style.display !== 'none') {
                displayNotifications(data.notifications);
            }
        }
    } catch (error) {
        console.error('Ошибка при проверке уведомлений:', error);
    }
}

// Функция для отображения уведомлений
function displayNotifications(notifications) {
    const notificationsContainer = document.getElementById('notifications-list');
    if (!notificationsContainer) return;

    notificationsContainer.innerHTML = '';

    if (!notifications || notifications.length === 0) {
        notificationsContainer.innerHTML = '<div class="no-data">Нет новых уведомлений</div>';
        return;
    }

    notifications.forEach(notification => {
        const notificationElement = document.createElement('div');
        notificationElement.className = 'notification-item';
        if (!notification.is_read) {
            notificationElement.classList.add('unread');
        }

        notificationElement.innerHTML = `
            <div class="notification-content">
                <p>${notification.message}</p>
                <span class="notification-time">${new Date(notification.created_at).toLocaleString()}</span>
            </div>
            ${notification.type === 'incoming_request' ? `
                <div class="notification-actions">
                    <button class="accept-btn" data-id="${notification.id}">Принять</button>
                    <button class="reject-btn" data-id="${notification.id}">Отклонить</button>
                </div>
            ` : ''}
            ${!notification.is_read ? '<div class="unread-dot"></div>' : ''}
        `;

        // Добавляем обработчики для кнопок
        if (notification.type === 'incoming_request') {
            const acceptBtn = notificationElement.querySelector('.accept-btn');
            const rejectBtn = notificationElement.querySelector('.reject-btn');

            acceptBtn.addEventListener('click', () => acceptRequest(notification.id));
            rejectBtn.addEventListener('click', () => rejectRequest(notification.id));
        }

        notificationsContainer.appendChild(notificationElement);
    });
}

// Функция для принятия запроса
async function acceptRequest(requestId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/accept-request`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({ request_id: requestId })
        });

        if (response.ok) {
            showNotification('Запрос принят', 'success');
            checkNotifications(); // Обновляем список уведомлений
            loadChats(); // Обновляем список чатов
        } else {
            const data = await response.json();
            showNotification(data.error || 'Ошибка при принятии запроса', 'error');
        }
    } catch (error) {
        console.error('Ошибка при принятии запроса:', error);
        showNotification('Ошибка при принятии запроса', 'error');
    }
}

// Функция для отклонения запроса
async function rejectRequest(requestId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/reject-request`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({ request_id: requestId })
        });

        if (response.ok) {
            showNotification('Запрос отклонен', 'success');
            checkNotifications(); // Обновляем список уведомлений
        } else {
            const data = await response.json();
            showNotification(data.error || 'Ошибка при отклонении запроса', 'error');
        }
    } catch (error) {
        console.error('Ошибка при отклонении запроса:', error);
        showNotification('Ошибка при отклонении запроса', 'error');
    }
}

// Добавляем периодическую проверку уведомлений
setInterval(checkNotifications, 30000); // Проверяем каждые 30 секунд

// Проверяем уведомления при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    if (localStorage.getItem('token')) {
        checkNotifications();
    }
});

// Обработчик клика по кнопке уведомлений
document.getElementById('notifications-btn').addEventListener('click', async () => {
    if (localStorage.getItem('token')) {
        // Показываем модальное окно
        const notificationsModal = document.getElementById('notifications-modal');
        notificationsModal.style.display = 'block';

        // Загружаем и отображаем уведомления
        try {
            const response = await fetch(`${API_BASE_URL}/api/notifications`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                }
            });

            if (response.ok) {
                const data = await response.json();
                displayNotifications(data.notifications);
            } else {
                showNotification('Ошибка при загрузке уведомлений', 'error');
            }
        } catch (error) {
            console.error('Ошибка при загрузке уведомлений:', error);
            showNotification('Ошибка при загрузке уведомлений', 'error');
        }
    }
});

// Добавляем обработчик для закрытия модального окна уведомлений
document.querySelector('#notifications-modal .close').addEventListener('click', () => {
    const notificationsModal = document.getElementById('notifications-modal');
    notificationsModal.style.display = 'none';
});

// Функция для загрузки чатов
async function loadChats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/chats`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });

        if (response.ok) {
            const data = await response.json();
            displayChats(data.chats);
        } else {
            showNotification('Ошибка при загрузке чатов', 'error');
        }
    } catch (error) {
        console.error('Ошибка при загрузке чатов:', error);
        showNotification('Ошибка при загрузке чатов', 'error');
    }
}

// Функция для отображения чатов
function displayChats(chats) {
    const chatsContainer = document.getElementById('chats-list');
    if (!chatsContainer) return;

    chatsContainer.innerHTML = '';

    if (!chats || chats.length === 0) {
        chatsContainer.innerHTML = '<div class="no-data">Нет активных чатов</div>';
        return;
    }

    chats.forEach(chat => {
        const chatElement = document.createElement('div');
        chatElement.className = 'chat-item';
        
        chatElement.innerHTML = `
            <div class="chat-info">
                <h3>@${chat.user_telegram}</h3>
                <p>${chat.type === 'incoming' ? 'Входящий' : 'Исходящий'} чат</p>
            </div>
            <div class="chat-meta">
                <span class="chat-time">${new Date(chat.created_at).toLocaleString()}</span>
            </div>
        `;

        chatsContainer.appendChild(chatElement);
    });
}

// Обработчик клика по кнопке чатов
document.getElementById('chats-btn').addEventListener('click', async () => {
    if (localStorage.getItem('token')) {
        // Показываем модальное окно чатов
        const chatsModal = document.getElementById('chats-modal');
        chatsModal.style.display = 'block';

        // Загружаем чаты
        await loadChats();
    }
});

// Добавляем обработчик для закрытия модального окна чатов
document.querySelector('#chats-modal .close').addEventListener('click', () => {
    const chatsModal = document.getElementById('chats-modal');
    chatsModal.style.display = 'none';
});