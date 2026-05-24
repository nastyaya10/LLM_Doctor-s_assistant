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
    let audioContext = null;
    let processorNode = null;
    let sourceNode = null;
    let pcmChunks = [];
    let pcmSampleRate = 16000;
    let sttConfig = null;

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

    function createAssistantMessageElement(payload) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        const avatar = document.createElement('div');
        avatar.className = 'avatar-icon';
        avatar.textContent = '🩺';

        const bubble = document.createElement('div');
        bubble.className = 'bubble assistant-rich';

        const answerBlock = document.createElement('div');
        answerBlock.className = 'answer-block';
        renderInlineMarkdown(answerBlock, payload.reply || '');
        bubble.appendChild(answerBlock);

        const sources = Array.isArray(payload.sources) ? payload.sources : [];
        if (sources.length > 0) {
            bubble.appendChild(createSourcesBlock(sources));
        }

        const debugChunks = Array.isArray(payload.debug_chunks) ? payload.debug_chunks : [];
        if (debugChunks.length > 0) {
            bubble.appendChild(createDebugChunksBlock(debugChunks));
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubble);
        return messageDiv;
    }

    function renderInlineMarkdown(container, text) {
        container.textContent = '';
        const parts = String(text).split(/(\*\*[^*]+\*\*)/g);

        parts.forEach((part) => {
            if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
                const strong = document.createElement('strong');
                strong.textContent = part.slice(2, -2);
                container.appendChild(strong);
            } else if (part) {
                container.appendChild(document.createTextNode(part));
            }
        });
    }

    function createSourcesBlock(sources) {
        const section = document.createElement('div');
        section.className = 'sources-block';

        const title = document.createElement('div');
        title.className = 'section-title';
        title.textContent = 'Источники';
        section.appendChild(title);

        const list = document.createElement('ul');
        list.className = 'sources-list';

        sources.forEach((sourceItem) => {
            const item = document.createElement('li');
            item.textContent = sourceItem.source || 'Без названия';
            list.appendChild(item);
        });

        section.appendChild(list);
        return section;
    }

    function createDebugChunksBlock(chunks) {
        const details = document.createElement('details');
        details.className = 'debug-chunks';

        const summary = document.createElement('summary');
        summary.textContent = `Найденные чанки (${chunks.length})`;
        details.appendChild(summary);

        chunks.forEach((chunk) => {
            const card = document.createElement('div');
            card.className = 'debug-chunk-card';

            const meta = document.createElement('div');
            meta.className = 'debug-chunk-meta';
            const scoreParts = [];
            if (chunk.score !== null && chunk.score !== undefined) scoreParts.push(`score ${chunk.score}`);
            if (chunk.rerank_score !== null && chunk.rerank_score !== undefined) scoreParts.push(`rerank ${chunk.rerank_score}`);
            meta.textContent = [
                `#${chunk.rank || '?'}`,
                chunk.source || 'Без названия',
                `global ${chunk.global_chunk_index ?? '?'}`,
                `file ${chunk.file_chunk_index ?? '?'}`,
                ...scoreParts,
            ].join(' · ');

            const text = document.createElement('div');
            text.className = 'debug-chunk-text';
            text.textContent = chunk.text || '';

            card.appendChild(meta);
            card.appendChild(text);
            details.appendChild(card);
        });

        return details;
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

    function setVoiceIcon(svg) {
        voiceBtn.innerHTML = svg;
    }

    function updateVoiceButton() {
        voiceBtn.classList.toggle('recording', isRecording);
        voiceBtn.classList.toggle('transcribing', isTranscribing);
        voiceBtn.disabled = isWaitingForResponse || isTranscribing;

        if (isTranscribing) {
            setVoiceIcon(`
                <svg class="voice-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M21 12a9 9 0 1 1-3-6.7"></path>
                </svg>
            `);
            voiceBtn.setAttribute('aria-label', 'Идет распознавание речи');
            voiceBtn.title = 'Идет распознавание';
        } else if (isRecording) {
            setVoiceIcon(`
                <svg class="voice-icon voice-icon-fill" viewBox="0 0 24 24" aria-hidden="true">
                    <rect x="8" y="8" width="8" height="8" rx="1.5"></rect>
                </svg>
            `);
            voiceBtn.setAttribute('aria-label', 'Остановить запись голоса');
            voiceBtn.title = 'Остановить запись';
        } else {
            setVoiceIcon(`
                <svg class="voice-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3Z"></path>
                    <path d="M19 11a7 7 0 0 1-14 0"></path>
                    <path d="M12 18v3"></path>
                    <path d="M8 21h8"></path>
                </svg>
            `);
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
        return response.json();
    }

    async function getSttConfig() {
        if (sttConfig) return sttConfig;
        try {
            const response = await fetch('/api/stt-config');
            sttConfig = response.ok ? await response.json() : { provider: 'openai' };
        } catch (error) {
            sttConfig = { provider: 'openai' };
        }
        return sttConfig;
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

    async function transcribePcmAudio(audioBytes, sampleRate) {
        const formData = new FormData();
        formData.append('audio', new Blob([audioBytes], { type: 'application/octet-stream' }), 'voice-query.raw');
        formData.append('audio_format', 'lpcm');
        formData.append('sample_rate', String(sampleRate));

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

    function startMediaRecorder(stream) {
        if (!window.MediaRecorder) {
            throw new Error('Браузер не поддерживает запись аудио.');
        }

        audioChunks = [];
        const mimeType = getSupportedMimeType();
        mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

        mediaRecorder.addEventListener('dataavailable', (event) => {
            if (event.data && event.data.size > 0) {
                audioChunks.push(event.data);
            }
        });

        mediaRecorder.addEventListener('stop', async () => {
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            await finishTranscription(() => transcribeAudio(audioBlob), () => {
                audioChunks = [];
                mediaRecorder = null;
            });
        });

        mediaRecorder.start();
    }

    function startPcmRecording(stream) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) {
            throw new Error('Браузер не поддерживает запись PCM-аудио.');
        }

        audioContext = new AudioContextClass({ sampleRate: 48000 });
        pcmSampleRate = 16000;
        pcmChunks = [];
        sourceNode = audioContext.createMediaStreamSource(stream);
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);

        processorNode.onaudioprocess = (event) => {
            if (!isRecording) return;
            const input = event.inputBuffer.getChannelData(0);
            pcmChunks.push(new Float32Array(input));
        };

        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);
    }

    async function stopPcmRecording() {
        const chunks = pcmChunks;
        const sourceSampleRate = audioContext ? audioContext.sampleRate : pcmSampleRate;
        cleanupPcmRecorder();
        const samples = mergeFloat32Arrays(chunks);
        const pcmBytes = encodePcm16(resampleAudio(samples, sourceSampleRate, pcmSampleRate));

        await finishTranscription(() => transcribePcmAudio(pcmBytes, pcmSampleRate), () => {
            pcmChunks = [];
        });
    }

    async function finishTranscription(transcribeFn, cleanupFn) {
        stopActiveStream();
        isRecording = false;
        isTranscribing = true;
        updateVoiceButton();
        setVoiceStatus('Распознаю речь...');

        try {
            const text = await transcribeFn();
            appendTranscription(text);
            setVoiceStatus(text ? 'Текст добавлен в поле ввода.' : 'Речь не распознана, попробуйте еще раз.', !text);
        } catch (error) {
            setVoiceStatus(error.message || 'Не удалось распознать речь.', true);
        } finally {
            cleanupFn();
            isTranscribing = false;
            updateVoiceButton();
        }
    }

    function cleanupPcmRecorder() {
        if (processorNode) {
            processorNode.disconnect();
            processorNode.onaudioprocess = null;
            processorNode = null;
        }
        if (sourceNode) {
            sourceNode.disconnect();
            sourceNode = null;
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
    }

    function mergeFloat32Arrays(chunks) {
        const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const merged = new Float32Array(totalLength);
        let offset = 0;
        chunks.forEach((chunk) => {
            merged.set(chunk, offset);
            offset += chunk.length;
        });
        return merged;
    }

    function resampleAudio(samples, sourceRate, targetRate) {
        if (sourceRate === targetRate) return samples;
        const ratio = sourceRate / targetRate;
        const length = Math.round(samples.length / ratio);
        const result = new Float32Array(length);

        for (let i = 0; i < length; i += 1) {
            const sourceIndex = i * ratio;
            const left = Math.floor(sourceIndex);
            const right = Math.min(left + 1, samples.length - 1);
            const fraction = sourceIndex - left;
            result[i] = samples[left] + (samples[right] - samples[left]) * fraction;
        }

        return result;
    }

    function encodePcm16(samples) {
        const buffer = new ArrayBuffer(samples.length * 2);
        const view = new DataView(buffer);
        samples.forEach((sample, index) => {
            const clipped = Math.max(-1, Math.min(1, sample));
            const value = clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff;
            view.setInt16(index * 2, value, true);
        });
        return buffer;
    }

    async function startVoiceRecording() {
        if (!window.isSecureContext) {
            setVoiceStatus('Микрофон доступен только по HTTPS или через localhost.', true);
            return;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setVoiceStatus('Браузер не поддерживает запись аудио.', true);
            return;
        }

        try {
            setVoiceStatus('Запрашиваю доступ к микрофону...');
            const config = await getSttConfig();
            activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            isRecording = true;
            if (config.provider === 'yandex') {
                startPcmRecording(activeStream);
            } else {
                startMediaRecorder(activeStream);
            }

            setVoiceStatus('Идет запись. Нажмите кнопку еще раз, чтобы остановить.');
            updateVoiceButton();
        } catch (error) {
            stopActiveStream();
            cleanupPcmRecorder();
            isRecording = false;
            updateVoiceButton();

            if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
                setVoiceStatus('Доступ к микрофону запрещен. Разрешите доступ в настройках браузера.', true);
            } else if (error.message) {
                setVoiceStatus(error.message, true);
            } else {
                setVoiceStatus('Не удалось начать запись с микрофона.', true);
            }
        }
    }

    function stopVoiceRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            setVoiceStatus('Останавливаю запись...');
        } else if (audioContext) {
            stopPcmRecording();
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
            const payload = await getAIResponse(trimmed);
            removeTypingIndicator();
            const assistantMsg = createAssistantMessageElement(payload);
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
