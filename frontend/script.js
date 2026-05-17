(function() {
    const messagesContainer = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const quickBtns = document.querySelectorAll('.quick-btn');

    let isWaitingForResponse = false;

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function createMessageElement(role, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'avatar-icon';
        avatar.textContent = role === 'user' ? '👤' : '🩺';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = text;

        if (role === 'user') {
            messageDiv.appendChild(bubble);
            messageDiv.appendChild(avatar);
        } else {
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(bubble);
        }

        return messageDiv;
    }

    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.id = 'typingIndicator';
        typingDiv.innerHTML = `
            <div class="avatar-icon">🩺</div>
            <div class="bubble" style="padding: 14px 18px;">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        messagesContainer.appendChild(typingDiv);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) indicator.remove();
    }

    async function getAIResponse(message) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        if (!response.ok) throw new Error('Ошибка сервера');
        const data = await response.json();
        return data.reply;
    }

    async function sendMessage(text) {
        const trimmed = text.trim();
        if (!trimmed || isWaitingForResponse) return;

        // Сообщение пользователя
        const userMsg = createMessageElement('user', trimmed);
        messagesContainer.appendChild(userMsg);
        scrollToBottom();
        messageInput.value = '';

        // Блокируем ввод
        isWaitingForResponse = true;
        sendBtn.disabled = true;
        messageInput.disabled = true;
        addTypingIndicator();

        try {
            const reply = await getAIResponse(trimmed);
            removeTypingIndicator();
            const assistantMsg = createMessageElement('assistant', reply);
            messagesContainer.appendChild(assistantMsg);
            scrollToBottom();
        } catch (error) {
            removeTypingIndicator();
            const errorMsg = createMessageElement('assistant', '⚠️ Не удалось получить ответ от модели. Проверьте сервер.');
            messagesContainer.appendChild(errorMsg);
            scrollToBottom();
        } finally {
            isWaitingForResponse = false;
            sendBtn.disabled = false;
            messageInput.disabled = false;
            messageInput.focus();
        }
    }

    sendBtn.addEventListener('click', () => {
        sendMessage(messageInput.value);
    });

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(messageInput.value);
        }
    });

    quickBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.getAttribute('data-query');
            if (query && !isWaitingForResponse) {
                sendMessage(query);
            }
        });
    });

    messageInput.focus();
    scrollToBottom();
})();