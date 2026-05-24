import http.server
import socketserver
import webbrowser
import json
import os
import sys
import tempfile
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

            self.send_json({"reply": reply})
        elif self.path == '/api/transcribe':
            self.handle_transcribe()
        else:
            self.send_error(404)

    def handle_transcribe(self):
        content_type = self.headers.get('Content-Type', '')
        if not content_type.startswith('multipart/form-data'):
            self.send_json({"error": "Ожидался multipart/form-data с полем audio."}, status=400)
            return

        audio_part = self.get_multipart_file("audio")
        if audio_part is None:
            self.send_json({"error": "Аудиофайл не найден в поле audio."}, status=400)
            return

        try:
            text = transcribe_audio_file(
                filename=audio_part.get_filename() or "voice-query.webm",
                audio_bytes=audio_part.get_payload(decode=True) or b"",
            )
        except RuntimeError as e:
            self.send_json({"error": str(e)}, status=400)
            return
        except Exception as e:
            self.send_json({"error": f"Ошибка распознавания речи: {str(e)}"}, status=500)
            return

        self.send_json({"text": text})

    def get_multipart_file(self, field_name: str):
        content_length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(content_length)
        raw_message = (
            f"Content-Type: {self.headers.get('Content-Type')}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode('utf-8') + body
        message = BytesParser(policy=policy.default).parsebytes(raw_message)

        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") == field_name:
                return part
        return None


def transcribe_audio_file(filename: str, audio_bytes: bytes) -> str:
    if not audio_bytes:
        raise RuntimeError("Получен пустой аудиофайл.")

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
