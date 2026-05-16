(function() {
    const messagesContainer = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    let isWaiting = false;

    function scrollToBottom() { messagesContainer.scrollTop = messagesContainer.scrollHeight; }

    function createMessage(role, text) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        div.innerHTML = role === 'user'
            ? `<div class="bubble">${text}</div><div class="avatar-icon">👤</div>`
            : `<div class="avatar-icon">🩺</div><div class="bubble">${text}</div>`;
        return div;
    }

    function addTyping() {
        const typing = document.createElement('div');
        typing.className = 'message assistant';
        typing.id = 'typingIndicator';
        typing.innerHTML = `<div class="avatar-icon">🩺</div><div class="bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
        messagesContainer.appendChild(typing);
        scrollToBottom();
    }

    function removeTyping() {
        const t = document.getElementById('typingIndicator');
        if (t) t.remove();
    }

    function generateReply(msg) {
        const lower = msg.toLowerCase();
        if (lower.includes('дифференциальный диагноз') || lower.includes('боль в грудной'))
            return '🔍 Диф.диагноз боли в грудной клетке:\n1. ОКС\n2. ТЭЛА\n3. Расслоение аорты\n4. Перикардит\n5. Пневмоторакс\nРекомендовано: ЭКГ, тропонины, D-димер, рентген ОГК.';
        if (lower.includes('оак') || lower.includes('лейкоцитоз'))
            return '🩸 ОАК: лейкоцитоз+нейтрофилез – вероятна бактериальная инфекция. Оцените сдвиг формулы, исключите стресс-лейкоцитоз. Назначьте СРБ, прокальцитонин.';
        if (lower.includes('антибиотик') || lower.includes('пневмония'))
            return '💊 Внебольничная пневмония:\n- Амоксициллин 1г 3р/сут\n- или макролид\n- при риске резистентности – амоксициллин/клавуланат.';
        if (lower.includes('дозировк') || lower.includes('хбп'))
            return '📋 Дозировки при ХБП:\n- Метформин: СКФ<30 противопоказан\n- Эноксапарин: снижение дозы на 50% при СКФ<30\nПроверьте СКФ пациента.';
        if (lower.includes('привет') || lower.includes('здравствуй'))
            return 'Здравствуйте! Готов помочь с клиническими вопросами.';
        return '📌 Рекомендую:\n- Полный анамнез\n- Физикальное обследование\n- Лабораторные тесты\nУточните запрос.';
    }

    function sendMessage(text) {
        if (!text.trim() || isWaiting) return;
        messagesContainer.appendChild(createMessage('user', text));
        messageInput.value = '';
        scrollToBottom();
        isWaiting = true;
        sendBtn.disabled = true;
        messageInput.disabled = true;
        addTyping();
        setTimeout(() => {
            removeTyping();
            messagesContainer.appendChild(createMessage('assistant', generateReply(text)));
            scrollToBottom();
            isWaiting = false;
            sendBtn.disabled = false;
            messageInput.disabled = false;
            messageInput.focus();
        }, 1400);
    }

    sendBtn.addEventListener('click', () => sendMessage(messageInput.value));
    messageInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage(messageInput.value);
        }
    });

    messageInput.focus();
    scrollToBottom();
})();