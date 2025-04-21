#!/bin/bash

echo "🔄 Killing existing uvicorn process..."
pkill -f "uvicorn app:app"

echo "📥 Pulling latest code from GitHub..."
git pull origin akmal-dev

echo "📦 Installing Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt

echo "⬇️ Pulling Ollama models..."
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

echo "🚀 Restarting uvicorn server..."
nohup /workspace/nuii-rag-chatbot/venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 11436 > ~/nuii.log 2>&1 &

echo "✅ All done!"
echo "📄 Check logs: tail -f ~/nuii.log"
