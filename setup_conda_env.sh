#!/bin/bash

# DeepVoice STT Voice Splitter - Conda Environment Setup Script
# 이 스크립트는 프로젝트 실행에 필요한 Conda 환경을 자동으로 설정합니다.

set -e  # 오류 발생 시 스크립트 중단

echo "🚀 DeepVoice STT Voice Splitter - Conda 환경 설정을 시작합니다."
echo "========================================================="

# 현재 스크립트 위치 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Conda 환경 이름
ENV_NAME="deepvoice-stt"

echo "📋 현재 작업 디렉토리: $SCRIPT_DIR"
echo "🏷️  환경 이름: $ENV_NAME"

# 1. 기존 환경 삭제 (선택사항)
if conda env list | grep -q "^$ENV_NAME "; then
    read -p "⚠️  기존 '$ENV_NAME' 환경이 존재합니다. 삭제하고 다시 만드시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[YyㅇㅖE예]$ ]]; then
        echo "🗑️  기존 환경 삭제 중..."
        conda env remove --name "$ENV_NAME" -y
    else
        echo "❌ 설치가 취소되었습니다."
        exit 1
    fi
fi

# 2. conda-forge, pytorch 채널 추가
echo "🔧 Conda 채널 설정 중..."
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels nvidia

# 3. 환경 생성
if [ -f "conda_environment.yml" ]; then
    echo "📦 conda_environment.yml 파일로 환경 생성 중..."
    conda env create -f conda_environment.yml
else
    echo "⚠️  conda_environment.yml 파일이 없습니다. 기본 환경을 생성합니다."
    conda create --name "$ENV_NAME" python=3.10 -y
fi

# 4. 환경 활성화
echo "🔄 Conda 환경 활성화 중..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 5. pip로 추가 패키지 설치 (requirements.txt가 있는 경우)
if [ -f "requirements.txt" ]; then
    echo "📦 requirements.txt 추가 패키지 설치 중..."
    pip install -r requirements.txt
fi

# 6. PyTorch CUDA 버전 확인 및 재설치 (필요시)
echo "🔍 PyTorch CUDA 지원 확인 중..."
python -c "
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU 개수: {torch.cuda.device_count()}')
else:
    print('⚠️  CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.')
"

# 7. Whisper.cpp 빌드
echo "🔨 Whisper.cpp 빌드 중..."
if [ ! -d "whisper.cpp" ]; then
    echo "⚠️  whisper.cpp 디렉토리가 없습니다. Git submodule을 초기화합니다."
    git submodule update --init --recursive
fi

cd whisper.cpp
if [ ! -d "build" ]; then
    mkdir build
fi
cd build

# CUDA 지원으로 빌드 시도
if command -v nvcc &> /dev/null; then
    echo "🚀 CUDA 지원으로 Whisper.cpp 빌드 중..."
    cmake .. -DWHISPER_CUDA=ON
else
    echo "🔧 CPU 모드로 Whisper.cpp 빌드 중..."
    cmake ..
fi

make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cd "$SCRIPT_DIR"

# 8. 필수 디렉토리 생성
echo "📁 필수 디렉토리 생성 중..."
mkdir -p resources
mkdir -p CosyVoice/pretrained_models
mkdir -p gemma
mkdir -p split_audio

# 9. 모델 다운로드 안내
echo "========================================================="
echo "✅ Conda 환경 설정이 완료되었습니다!"
echo "========================================================="
echo
echo "🔧 다음 단계:"
echo "1. 환경 활성화: conda activate $ENV_NAME"
echo "2. 필수 모델 다운로드:"
echo "   - Whisper Large v3 Turbo (1.5GB)"
echo "   - CosyVoice2-0.5B (약 2GB)"
echo "   - Silero VAD (자동 다운로드)"
echo "   - Gemma 3 모델 (선택사항, 7.7GB 또는 16.4GB)"
echo
echo "3. 모델 다운로드 스크립트 실행:"
echo "   bash download_models.sh"
echo
echo "4. 프로그램 실행:"
echo "   python STT_Voice_Spliter.py"
echo
echo "📝 참고:"
echo "- PyCharm에서 이 환경을 사용하려면:"
echo "  Settings > Project > Python Interpreter에서"
echo "  'Add Interpreter > Conda Environment > Existing environment'"
echo "  선택 후 $(conda info --base)/envs/$ENV_NAME/bin/python 경로 설정"
echo
echo "🎉 설정 완료! 즐거운 개발 되세요!"