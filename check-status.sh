#!/bin/bash

echo "🔍 DeepVoice 시스템 상태 확인"
echo "============================="

cd docker

echo "📦 컨테이너 상태:"
docker compose ps

echo ""
echo "🏥 서비스 헬스체크:"
echo "전체 시스템: $(curl -s http://localhost:8000/health | jq -r .status 2>/dev/null || echo 'N/A')"
echo "Whisper STT: $(curl -s http://localhost:8001/health | jq -r .status 2>/dev/null || echo 'N/A')"
echo "Translator: $(curl -s http://localhost:8002/health | jq -r .status 2>/dev/null || echo 'N/A')"
echo "TTS: $(curl -s http://localhost:8003/health | jq -r .status 2>/dev/null || echo 'N/A')"
echo "Audio Processor: $(curl -s http://localhost:8005/health | jq -r .status 2>/dev/null || echo 'N/A')"

echo ""
echo "💾 디스크 사용량:"
du -sh ../output/ 2>/dev/null || echo "출력 폴더 없음"

echo ""
echo "🖥️ 시스템 리소스:"
echo "메모리: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
echo "디스크: $(df -h . | tail -1 | awk '{print $3 "/" $2 " (" $5 " 사용)"}')"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🎮 GPU 상태:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk '{print "GPU: " $1 "%, VRAM: " $2 "/" $3 " MB"}'
fi
