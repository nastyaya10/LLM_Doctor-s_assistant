import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Указываем директорию с фронтендом (по умолчанию – текущая папка)
DIRECTORY = Path(__file__).parent


class MyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)


if __name__ == '__main__':
    PORT = 8000
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Фронтенд доступен по адресу: http://localhost:{PORT}")
        webbrowser.open(f"http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nСервер остановлен.")
