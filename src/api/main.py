import http.server
import socketserver
import webbrowser
import json
import os
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ИМПОРТЫ ИЗ НОВОЙ СТРУКТУРЫ
from src.services.md_agent import MedicalAgent

# Путь к фронтенду
FRONTEND_DIR = PROJECT_ROOT / "webui"

agents_by_session = {}


def get_agent(session_id: str) -> MedicalAgent:
    if session_id not in agents_by_session:
        print(f"[Session] Создан новый агент для session_id={session_id}")
        agents_by_session[session_id] = MedicalAgent()
    return agents_by_session[session_id]


class ChatAPIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            user_message = data.get('message', '')
            session_id = data.get('session_id')

            if not session_id:
                self.send_response(400)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                response = json.dumps({"reply": "Ошибка: не передан session_id."}, ensure_ascii=False)
                self.wfile.write(response.encode('utf-8'))
                return

            try:
                session_agent = get_agent(session_id)
                reply = session_agent.generate_response(user_message)
            except Exception as e:
                reply = f"Ошибка при обращении к модели: {str(e)}"

            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            response = json.dumps({"reply": reply}, ensure_ascii=False)
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)


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
