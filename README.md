Для запуска:

pip install -r requirements.txt
echo 'OPENAI_API_KEY=sk-ваш-ключ' > .env
echo 'OPENAI_BASE_URL=https://aipipe.org/openai/v1' >> .env
echo 'MD_FILE=sample.md' >> .env
python main.py