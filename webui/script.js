(function() {
    const messagesContainer = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const voiceBtn = document.getElementById('voiceBtn');
    const voiceStatus = document.getElementById('voiceStatus');
    const quickBtns = document.querySelectorAll('.quick-btn');

    let isWaitingForResponse = false;
    let isRecording = false;
    let isTranscribing = false;
    let mediaRecorder = null;
    let audioChunks = [];
    let activeStream = null;

    function getSessionId() {
        let sessionId = localStorage.getItem('medai_session_id');
        if (!sessionId) {
            if (window.crypto && crypto.randomUUID) {
                sessionId = crypto.randomUUID();
            } else {
                sessionId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
            }
            localStorage.setItem('medai_session_id', sessionId);
        }
        return sessionId;
    }

    const sessionId = getSessionId();

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

    function setVoiceStatus(text, isError = false) {
        voiceStatus.textContent = text;
        voiceStatus.classList.toggle('error', isError);
    }

    function updateVoiceButton() {
        voiceBtn.classList.toggle('recording', isRecording);
        voiceBtn.classList.toggle('transcribing', isTranscribing);
        voiceBtn.disabled = isWaitingForResponse || isTranscribing;

        if (isTranscribing) {
            voiceBtn.textContent = '…';
            voiceBtn.setAttribute('aria-label', 'Идет распознавание речи');
            voiceBtn.title = 'Идет распознавание';
        } else if (isRecording) {
            voiceBtn.textContent = '■';
            voiceBtn.setAttribute('aria-label', 'Остановить запись голоса');
            voiceBtn.title = 'Остановить запись';
        } else {
            voiceBtn.textContent = '🎙️';
            voiceBtn.setAttribute('aria-label', 'Начать запись голоса');
            voiceBtn.title = 'Начать запись';
        }
    }

    async function getAIResponse(message) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                message
            })
        });
        if (!response.ok) throw new Error('Ошибка сервера');
        const data = await response.json();
        return data.reply;
    }

    async function transcribeAudio(audioBlob) {
        const formData = new FormData();
        const extension = audioBlob.type.includes('mp4') ? 'mp4' : 'webm';
        formData.append('audio', audioBlob, `voice-query.${extension}`);

        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(data.error || 'Не удалось распознать речь');
        }
        return data.text || '';
    }

    function appendTranscription(text) {
        const recognizedText = text.trim();
        if (!recognizedText) return;

        // Не перезаписываем уже набранный клинический контекст: голосовой текст добавляется в конец черновика.
        const currentText = messageInput.value.trim();
        messageInput.value = currentText ? `${currentText} ${recognizedText}` : recognizedText;
        messageInput.focus();
    }

    function getSupportedMimeType() {
        if (!window.MediaRecorder || !MediaRecorder.isTypeSupported) return '';
        const candidates = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4'
        ];
        return candidates.find(type => MediaRecorder.isTypeSupported(type)) || '';
    }

    function stopActiveStream() {
        if (activeStream) {
            activeStream.getTracks().forEach(track => track.stop());
            activeStream = null;
        }
    }

    async function startVoiceRecording() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
            setVoiceStatus('Браузер не поддерживает запись аудио.', true);
            return;
        }

        try {
            setVoiceStatus('Запрашиваю доступ к микрофону...');
            activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];

            const mimeType = getSupportedMimeType();
            mediaRecorder = new MediaRecorder(activeStream, mimeType ? { mimeType } : undefined);

            mediaRecorder.addEventListener('dataavailable', (event) => {
                if (event.data && event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            });

            mediaRecorder.addEventListener('stop', async () => {
                const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                stopActiveStream();
                isRecording = false;
                isTranscribing = true;
                updateVoiceButton();
                setVoiceStatus('Распознаю речь...');

                try {
                    const text = await transcribeAudio(audioBlob);
                    appendTranscription(text);
                    setVoiceStatus(text ? 'Текст добавлен в поле ввода.' : 'Речь не распознана, попробуйте еще раз.', !text);
                } catch (error) {
                    setVoiceStatus(error.message || 'Не удалось распознать речь.', true);
                } finally {
                    audioChunks = [];
                    mediaRecorder = null;
                    isTranscribing = false;
                    updateVoiceButton();
                }
            });

            mediaRecorder.start();
            isRecording = true;
            setVoiceStatus('Идет запись. Нажмите кнопку еще раз, чтобы остановить.');
            updateVoiceButton();
        } catch (error) {
            stopActiveStream();
            isRecording = false;
            updateVoiceButton();

            if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
                setVoiceStatus('Доступ к микрофону запрещен. Разрешите доступ в настройках браузера.', true);
            } else {
                setVoiceStatus('Не удалось начать запись с микрофона.', true);
            }
        }
    }

    function stopVoiceRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            setVoiceStatus('Останавливаю запись...');
        }
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
        updateVoiceButton();
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
            updateVoiceButton();
            messageInput.focus();
        }
    }

    sendBtn.addEventListener('click', () => {
        sendMessage(messageInput.value);
    });

    voiceBtn.addEventListener('click', () => {
        if (isRecording) {
            stopVoiceRecording();
        } else {
            startVoiceRecording();
        }
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
    updateVoiceButton();
    scrollToBottom();
})();
