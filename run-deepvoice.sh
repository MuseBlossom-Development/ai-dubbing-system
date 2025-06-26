#!/bin/bash

echo "🎙️ DeepVoice STT Voice Splitter 시작"
echo "====================================="

# 브랜치 확인
if [ "$(git branch --show-current)" != "docker-compose-migration" ]; then
    echo "📝 docker-compose-migration 브랜치로 전환 중..."
    git checkout docker-compose-migration
fi

# 디렉토리 확인
mkdir -p {input,output,temp,config}

# Docker Compose 실행
cd docker

echo ""
echo "🚀 실행 옵션을 선택하세요:"
echo "1) 립싱크 제외 버전 (권장, 안정적)"
echo "2) 전체 버전 (립싱크 포함)"
echo "3) 백그라운드 실행 (립싱크 제외)"
echo ""

read -p "선택 (1-3): " choice

case $choice in
    1)
        echo "🔄 립싱크 제외 버전으로 실행 중..."
        docker compose -f docker-compose.no-lipsync.yml up --build
        ;;
    2)
        echo "🔄 전체 버전으로 실행 중..."
        docker compose up --build
        ;;
    3)
        echo "🔄 백그라운드에서 실행 중..."
        docker compose -f docker-compose.no-lipsync.yml up -d --build
        echo ""
        echo "✅ 백그라운드 실행 시작!"
        echo "📊 상태 확인: docker compose ps"
        echo "📋 로그 확인: docker compose logs -f"
        echo "🛑 중지: docker compose down"
        echo "🌐 웹 UI: http://localhost:7860"
        ;;
    *)
        echo "❌ 잘못된 선택입니다."
        exit 1
        ;;
esac
