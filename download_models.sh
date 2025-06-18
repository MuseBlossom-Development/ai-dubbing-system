#!/bin/bash

# DeepVoice STT Voice Splitter - Model Download Script
# 필수 모델 파일들을 자동으로 다운로드합니다.

set -e  # 오류 발생 시 스크립트 중단

echo "📦 DeepVoice STT Voice Splitter - 모델 다운로드를 시작합니다."
echo "========================================================="

# 현재 스크립트 위치 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 필수 디렉토리 생성
mkdir -p resources
mkdir -p CosyVoice/pretrained_models
mkdir -p whisper.cpp/models
mkdir -p gemma

echo "📋 현재 작업 디렉토리: $SCRIPT_DIR"

# 다운로드 함수
download_with_progress() {
    local url="$1"
    local output="$2"
    local description="$3"
    
    echo "⬇️  다운로드 중: $description"
    echo "   URL: $url"
    echo "   저장 위치: $output"
    
    if command -v wget &> /dev/null; then
        wget --progress=bar:force:noscroll -O "$output" "$url"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$output" "$url"
    else
        echo "❌ wget 또는 curl이 필요합니다."
        exit 1
    fi
    
    echo "✅ 다운로드 완료: $description"
    echo
}

# 파일 존재 확인 함수
check_file() {
    local file="$1"
    local name="$2"
    
    if [ -f "$file" ]; then
        local size=$(du -h "$file" | cut -f1)
        echo "✅ $name: $file ($size)"
        return 0
    else
        echo "❌ $name: $file (없음)"
        return 1
    fi
}

echo "🔍 기존 모델 파일 확인 중..."
echo "========================================================="

# 1. Whisper Large v3 Turbo 모델 (필수)
WHISPER_MODEL="resources/ggml-large-v3-turbo.bin"
WHISPER_URL="https://huggingface.co/ggml-org/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"

if ! check_file "$WHISPER_MODEL" "Whisper Large v3 Turbo"; then
    echo "📥 Whisper Large v3 Turbo 모델 다운로드 중... (약 1.5GB)"
    download_with_progress "$WHISPER_URL" "$WHISPER_MODEL" "Whisper Large v3 Turbo"
fi

# 2. Silero VAD 모델 (필수)
SILERO_MODEL="whisper.cpp/models/ggml-silero-v5.1.2.bin"
SILERO_URL="https://huggingface.co/ggml-org/whisper.cpp/resolve/main/ggml-silero-v5.1.2.bin"

if ! check_file "$SILERO_MODEL" "Silero VAD v5.1.2"; then
    echo "📥 Silero VAD 모델 다운로드 중... (약 39MB)"
    download_with_progress "$SILERO_URL" "$SILERO_MODEL" "Silero VAD v5.1.2"
fi

# 3. CosyVoice2 모델 다운로드 (Python 스크립트 사용)
COSYVOICE_MODEL_DIR="CosyVoice/pretrained_models/CosyVoice2-0.5B"
COSYVOICE_MAIN_FILE="$COSYVOICE_MODEL_DIR/llm.pt"

if ! check_file "$COSYVOICE_MAIN_FILE" "CosyVoice2-0.5B"; then
    echo "📥 CosyVoice2-0.5B 모델 다운로드 중... (약 2GB)"
    echo "   ModelScope를 통해 다운로드합니다..."
    
    python3 << 'EOF'
import os
import sys
try:
    from modelscope import snapshot_download
    print("📦 ModelScope를 사용하여 CosyVoice2 모델을 다운로드합니다...")
    snapshot_download(
        'iic/CosyVoice2-0.5B', 
        local_dir='CosyVoice/pretrained_models/CosyVoice2-0.5B'
    )
    print("✅ CosyVoice2 모델 다운로드 완료")
except ImportError:
    print("❌ modelscope 패키지가 설치되지 않았습니다.")
    print("   pip install modelscope 명령으로 설치해주세요.")
    sys.exit(1)
except Exception as e:
    print(f"❌ CosyVoice2 모델 다운로드 오류: {e}")
    print("🔧 수동 다운로드가 필요합니다:")
    print("   https://huggingface.co/iic/CosyVoice2-0.5B")
    sys.exit(1)
EOF
    echo
fi

# 4. Gemma 3 모델 (선택사항)
echo "🤖 Gemma 3 모델 다운로드 (선택사항)"
echo "========================================================="

GEMMA_12B="gemma/gemma-3-12b-it-q4_0.gguf"
GEMMA_27B="gemma/gemma-3-27b-it-q4_0.gguf"

echo "Gemma 3 모델은 번역 기능에 사용됩니다."
echo "1) Gemma 3 12B (7.7GB) - 빠른 번역"
echo "2) Gemma 3 27B (16.4GB) - 고품질 번역"
echo "3) 건너뛰기"
echo

read -p "어떤 모델을 다운로드하시겠습니까? (1/2/3): " -n 1 -r
echo

case $REPLY in
    1)
        if ! check_file "$GEMMA_12B" "Gemma 3 12B"; then
            echo "📥 Gemma 3 12B 모델 다운로드 중... (약 7.7GB)"
            echo "⚠️  주의: 이 다운로드는 Hugging Face 계정이 필요할 수 있습니다."
            
            GEMMA_12B_URL="https://huggingface.co/bartowski/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-q4_0.gguf"
            
            if command -v huggingface-cli &> /dev/null; then
                echo "🔧 Hugging Face CLI를 사용하여 다운로드합니다..."
                huggingface-cli download bartowski/gemma-3-12b-it-GGUF gemma-3-12b-it-q4_0.gguf --local-dir gemma
            else
                echo "🔧 wget/curl을 사용하여 다운로드를 시도합니다..."
                download_with_progress "$GEMMA_12B_URL" "$GEMMA_12B" "Gemma 3 12B"
            fi
        fi
        ;;
    2)
        if ! check_file "$GEMMA_27B" "Gemma 3 27B"; then
            echo "📥 Gemma 3 27B 모델 다운로드 중... (약 16.4GB)"
            echo "⚠️  주의: 이 다운로드는 Hugging Face 계정이 필요할 수 있습니다."
            
            GEMMA_27B_URL="https://huggingface.co/bartowski/gemma-3-27b-it-GGUF/resolve/main/gemma-3-27b-it-q4_0.gguf"
            
            if command -v huggingface-cli &> /dev/null; then
                echo "🔧 Hugging Face CLI를 사용하여 다운로드합니다..."
                huggingface-cli download bartowski/gemma-3-27b-it-GGUF gemma-3-27b-it-q4_0.gguf --local-dir gemma
            else
                echo "🔧 wget/curl을 사용하여 다운로드를 시도합니다..."
                download_with_progress "$GEMMA_27B_URL" "$GEMMA_27B" "Gemma 3 27B"
            fi
        fi
        ;;
    3)
        echo "⏭️  Gemma 3 모델 다운로드를 건너뜁니다."
        echo "   나중에 번역 기능이 필요하면 이 스크립트를 다시 실행하세요."
        ;;
    *)
        echo "❌ 잘못된 선택입니다. Gemma 3 모델 다운로드를 건너뜁니다."
        ;;
esac

echo
echo "🔍 최종 모델 파일 확인"
echo "========================================================="

# 최종 확인
check_file "$WHISPER_MODEL" "Whisper Large v3 Turbo"
check_file "$SILERO_MODEL" "Silero VAD v5.1.2"
check_file "$COSYVOICE_MAIN_FILE" "CosyVoice2-0.5B"

if [ -f "$GEMMA_12B" ]; then
    check_file "$GEMMA_12B" "Gemma 3 12B"
elif [ -f "$GEMMA_27B" ]; then
    check_file "$GEMMA_27B" "Gemma 3 27B"
else
    echo "⚪ Gemma 3 모델: 없음 (번역 기능 제한)"
fi

echo
echo "========================================================="
echo "✅ 모델 다운로드가 완료되었습니다!"
echo "========================================================="
echo
echo "🚀 이제 프로그램을 실행할 수 있습니다:"
echo "   conda activate deepvoice-stt"
echo "   python STT_Voice_Spliter.py"
echo
echo "📝 참고사항:"
echo "- 모든 모델이 정상적으로 다운로드되었는지 확인하세요"
echo "- Gemma 3 모델이 없으면 번역 기능이 제한됩니다"
echo "- GPU를 사용하려면 CUDA가 설치되어 있어야 합니다"
echo
echo "🎉 설정 완료! 즐거운 사용 되세요!"