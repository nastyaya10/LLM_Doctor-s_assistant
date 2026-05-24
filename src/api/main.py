import http.server
import socketserver
import webbrowser
import json
import os
import ssl
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from email import policy
from email.parser import BytesParser
from pathlib import Path

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ИМПОРТЫ ИЗ НОВОЙ СТРУКТУРЫ
from src.services.md_agent import MedicalAgent
from dotenv import load_dotenv
from openai import OpenAI

try:
    import certifi
except ImportError:
    certifi = None

# Путь к фронтенду
FRONTEND_DIR = PROJECT_ROOT / "webui"
load_dotenv(PROJECT_ROOT / ".env")

agents_by_session = {}


def get_agent(session_id: str) -> MedicalAgent:
    if session_id not in agents_by_session:
        print(f"[Session] Создан новый агент для session_id={session_id}")
        agents_by_session[session_id] = MedicalAgent()
    return agents_by_session[session_id]


class ChatAPIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def send_json(self, payload, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        response = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(response.encode('utf-8'))

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            user_message = data.get('message', '')
            session_id = data.get('session_id')

            if not session_id:
                self.send_json({"reply": "Ошибка: не передан session_id."}, status=400)
                return

            try:
                session_agent = get_agent(session_id)
                reply = session_agent.generate_response(user_message)
            except Exception as e:
                reply = f"Ошибка при обращении к модели: {str(e)}"

            self.send_json(normalize_agent_reply(reply))
        elif self.path == '/api/transcribe':
            self.handle_transcribe()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == '/api/stt-config':
            self.send_json({"provider": get_stt_provider()})
            return
        super().do_GET()

    def handle_transcribe(self):
        content_type = self.headers.get('Content-Type', '')
        if not content_type.startswith('multipart/form-data'):
            self.send_json({"error": "Ожидался multipart/form-data с полем audio."}, status=400)
            return

        parts = self.get_multipart_parts()
        audio_part = parts.get("audio")
        if audio_part is None:
            self.send_json({"error": "Аудиофайл не найден в поле audio."}, status=400)
            return

        try:
            text = transcribe_audio_file(
                filename=audio_part.get_filename() or "voice-query.webm",
                audio_bytes=audio_part.get_payload(decode=True) or b"",
                audio_format=self.get_multipart_text(parts, "audio_format"),
                sample_rate=self.get_multipart_text(parts, "sample_rate"),
            )
        except RuntimeError as e:
            self.send_json({"error": str(e)}, status=400)
            return
        except Exception as e:
            self.send_json({"error": f"Ошибка распознавания речи: {str(e)}"}, status=500)
            return

        self.send_json({"text": text})

    def get_multipart_parts(self):
        content_length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(content_length)
        raw_message = (
            f"Content-Type: {self.headers.get('Content-Type')}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode('utf-8') + body
        message = BytesParser(policy=policy.default).parsebytes(raw_message)

        parts = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name:
                parts[name] = part
        return parts

    def get_multipart_text(self, parts, field_name: str):
        part = parts.get(field_name)
        if part is None:
            return None
        payload = part.get_payload(decode=True)
        if payload is None:
            return None
        return payload.decode(part.get_content_charset() or "utf-8", errors="replace").strip()


def get_stt_provider() -> str:
    explicit_provider = os.getenv("STT_PROVIDER", "").strip().lower()
    if explicit_provider in ("openai", "yandex"):
        return explicit_provider

    if os.getenv("STT_BASE_URL") or os.getenv("STT_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return "openai"

    base_url = os.getenv("BASE_URL", "")
    if "yandex" in base_url and os.getenv("API_KEY") and os.getenv("FOLDER_ID"):
        return "yandex"

    return "openai"


def normalize_agent_reply(reply):
    if not isinstance(reply, str):
        return {"reply": str(reply), "sources": [], "debug_chunks": []}

    try:
        payload = json.loads(reply)
    except json.JSONDecodeError:
        return {"reply": reply, "sources": [], "debug_chunks": []}

    if not isinstance(payload, dict):
        return {"reply": reply, "sources": [], "debug_chunks": []}

    answer = payload.get("answer")
    if not isinstance(answer, str):
        return {"reply": reply, "sources": [], "debug_chunks": []}

    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    debug_chunks = payload.get("debug_chunks", [])
    if not isinstance(debug_chunks, list):
        debug_chunks = []

    return {"reply": answer, "sources": sources, "debug_chunks": debug_chunks}


def transcribe_audio_file(filename: str, audio_bytes: bytes, audio_format: str | None = None, sample_rate: str | None = None) -> str:
    if not audio_bytes:
        raise RuntimeError("Получен пустой аудиофайл.")

    if get_stt_provider() == "yandex":
        return transcribe_yandex_speechkit(audio_bytes, audio_format=audio_format, sample_rate=sample_rate)

    api_key = os.getenv("STT_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("Не задан ключ для транскрибации: STT_API_KEY, OPENAI_API_KEY или API_KEY.")

    if os.getenv("STT_BASE_URL"):
        base_url = os.getenv("STT_BASE_URL")
    elif os.getenv("STT_API_KEY") or os.getenv("OPENAI_API_KEY"):
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    else:
        base_url = os.getenv("BASE_URL", "https://api.openai.com/v1")

    model = os.getenv("STT_MODEL", "whisper-1")
    suffix = Path(filename).suffix or ".webm"
    client = OpenAI(api_key=api_key, base_url=base_url)

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        with open(tmp.name, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio,
            )

    return getattr(transcription, "text", "") or ""


def transcribe_yandex_speechkit(audio_bytes: bytes, audio_format: str | None = None, sample_rate: str | None = None) -> str:
    api_key = os.getenv("STT_API_KEY") or os.getenv("API_KEY")
    folder_id = os.getenv("STT_FOLDER_ID") or os.getenv("FOLDER_ID")
    if not api_key:
        raise RuntimeError("Не задан ключ для Yandex SpeechKit: STT_API_KEY или API_KEY.")
    if not folder_id:
        raise RuntimeError("Не задан каталог для Yandex SpeechKit: STT_FOLDER_ID или FOLDER_ID.")

    speech_format = (audio_format or "lpcm").lower()
    params = {
        "lang": os.getenv("STT_LANG", "ru-RU"),
        "topic": os.getenv("STT_TOPIC", "general"),
        "format": speech_format,
        "folderId": folder_id,
    }
    if speech_format == "lpcm":
        params["sampleRateHertz"] = sample_rate or os.getenv("STT_SAMPLE_RATE", "16000")

    url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        data=audio_bytes,
        headers={
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/octet-stream",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30, context=get_ssl_context()) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Yandex SpeechKit вернул ошибку {e.code}: {extract_yandex_error(error_body)}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Не удалось подключиться к Yandex SpeechKit: {e.reason}") from e

    return payload.get("result", "") or ""


def get_ssl_context():
    if certifi is None:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def extract_yandex_error(error_body: str) -> str:
    try:
        payload = json.loads(error_body)
    except json.JSONDecodeError:
        return error_body or "пустой ответ"
    return payload.get("message") or payload.get("error") or error_body


if __name__ == '__main__':
    HOST = os.getenv("HOST", "")
    PORT = int(os.getenv("PORT", "8000"))
    OPEN_BROWSER = os.getenv("OPEN_BROWSER", "0").lower() in ("1", "true", "yes")

    if not FRONTEND_DIR.exists():
        print(f"Ошибка: папка {FRONTEND_DIR} не найдена.")
        sys.exit(1)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((HOST, PORT), ChatAPIHandler) as httpd:
        display_host = HOST or "localhost"
        print(f"Сервер с агентом запущен на http://{display_host}:{PORT}")
        if OPEN_BROWSER:
            webbrowser.open(f"http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nСервер остановлен.")
