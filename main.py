import http.server
import socketserver
import webbrowser
import json
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы импортировать AI.md_agent
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Импортируем вашего агента. Предполагается, что в md_agent.py есть класс MedicalAgent
# или функция generate_response. Настройте импорт под ваш интерфейс.
from AI.md_agent import MedicalAgent

# Инициализируем агента (если требуется загрузка модели, делаем это один раз при старте)
agent = MedicalAgent()  # Подставьте ваш способ инициализации

FRONTEND_DIR = PROJECT_ROOT / "frontend"


class ChatAPIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)
            user_message = data.get('message', '')

            try:
                # Вызов вашего агента – замените на реальный метод
                # Например: reply = agent.get_response(user_message)
                reply = agent.generate_response(user_message)
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
    PORT = 8000
    if not FRONTEND_DIR.exists():
        print(f"Ошибка: папка {FRONTEND_DIR} не найдена.")
        sys.exit(1)

    with socketserver.TCPServer(("", PORT), ChatAPIHandler) as httpd:
        print(f"Сервер с агентом запущен на http://localhost:{PORT}")
        webbrowser.open(f"http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nСервер остановлен.")
