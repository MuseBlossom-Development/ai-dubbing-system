# #!/bin/bash
# set -e  # 오류 발생 시 스크립트 종료

# echo "========== Miniconda 확인 =========="
# if ! command -v conda &> /dev/null; then
#     echo "Miniconda가 설치되어 있지 않습니다. Miniconda를 설치합니다..."
#     # Miniconda 설치 스크립트 다운로드
#     MINICONDA_SCRIPT="Miniconda3-latest-MacOSX-arm64.sh"
#     curl -L https://repo.anaconda.com/miniconda/$MINICONDA_SCRIPT -o $MINICONDA_SCRIPT
    
#     # 설치 실행
#     bash $MINICONDA_SCRIPT -b -p $HOME/miniconda3
    
#     # 환경 변수 설정
#     export PATH="$HOME/miniconda3/bin:$PATH"
    
#     # conda 초기화
#     eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    
#     # 설치 스크립트 삭제
#     rm $MINICONDA_SCRIPT
# else
#     echo "Miniconda가 이미 설치되어 있습니다."
#     # conda 활성화
#     eval "$(conda shell.bash hook)"
# fi

# echo "========== Conda 환경 생성 =========="
# # 이미 존재하는 경우 삭제하고 새로 생성
# conda env remove -n stt_env -y || true
# conda create -y -n stt_env python=3.11 tk ffmpeg

# echo "========== Conda 환경 활성화 =========="
# conda activate stt_env

# echo "========== Python 패키지 설치 =========="
# pip install aiohttp aiofiles requests torch torchaudio coremltools ane_transformers openai-whisper

# echo "========== 메인 스크립트 실행 =========="
# python STT_Voice_Spliter.py

# #!/bin/bash
# set -e  # 오류 발생 시 스크립트 종료

# echo "========== Miniconda 확인 =========="
# if ! command -v conda &> /dev/null; then
#     echo "Miniconda가 설치되어 있지 않습니다. Miniconda를 설치합니다..."
#     # Miniconda 설치 스크립트 다운로드
#     MINICONDA_SCRIPT="Miniconda3-latest-MacOSX-arm64.sh"
#     curl -L https://repo.anaconda.com/miniconda/$MINICONDA_SCRIPT -o $MINICONDA_SCRIPT
    
#     # 설치 실행
#     bash $MINICONDA_SCRIPT -b -p $HOME/miniconda3
    
#     # 환경 변수 설정
#     export PATH="$HOME/miniconda3/bin:$PATH"
    
#     # conda 초기화
#     eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    
#     # 설치 스크립트 삭제
#     rm $MINICONDA_SCRIPT
# else
#     echo "Miniconda가 이미 설치되어 있습니다."
#     # conda 활성화
#     eval "$(conda shell.bash hook)"
# fi

# echo "========== Conda 환경 생성 =========="
# # 이미 존재하는 경우 삭제하고 새로 생성
# conda env remove -n stt_env -y || true
# conda create -y -n stt_env python=3.11 tk ffmpeg

# echo "========== Conda 환경 활성화 =========="
# conda activate stt_env

# echo "========== Python 패키지 설치 =========="
# pip install aiohttp aiofiles requests torch torchaudio coremltools ane_transformers openai-whisper

# echo "========== 메인 스크립트 실행 =========="
# python STT_Voice_Spliter.py

#!/bin/bash
set -e  # 오류 발생 시 스크립트 종료

echo "========== Miniconda 확인 =========="
if ! command -v conda &> /dev/null; then
    echo "Miniconda가 설치되어 있지 않습니다. Miniconda를 설치합니다..."
    # 운영체제 확인 및 적절한 Miniconda 스크립트 선택
    if [[ $(uname -m) == "arm64" ]]; then
        MINICONDA_SCRIPT="Miniconda3-latest-MacOSX-arm64.sh"
    else
        MINICONDA_SCRIPT="Miniconda3-latest-MacOSX-x86_64.sh"
    fi
    
    curl -L https://repo.anaconda.com/miniconda/$MINICONDA_SCRIPT -o $MINICONDA_SCRIPT
    
    # 설치 실행
    bash $MINICONDA_SCRIPT -b -p $HOME/miniconda3
    
    # 환경 변수 설정
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # conda 초기화
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    
    # 설치 스크립트 삭제
    rm $MINICONDA_SCRIPT
else
    echo "Miniconda가 이미 설치되어 있습니다."
    # conda 활성화
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "========== Conda 환경 생성 =========="
# 안전한 환경 제거 및 생성 방법
if conda env list | grep -q "stt_env"; then
    echo "기존 stt_env 환경을 제거합니다..."
    conda deactivate || true  # 현재 활성화된 환경이 있으면 비활성화
    conda env remove -n stt_env --yes || echo "환경 제거 실패, 계속 진행합니다."
    # 폴더가 남아있다면 강제 삭제
    if [ -d "$(conda info --base)/envs/stt_env" ]; then
        echo "환경 폴더가 남아있어 수동으로 제거합니다..."
        rm -rf "$(conda info --base)/envs/stt_env"
    fi
fi

echo "새로운 stt_env 환경을 생성합니다..."
conda create -y -n stt_env python=3.11 tk ffmpeg

echo "========== Conda 환경 활성화 =========="
conda activate stt_env || { echo "환경 활성화 실패"; exit 1; }
echo "현재 활성화된 환경: $(conda info --envs | grep '*')"

echo "========== 시스템 종속성 설치 =========="
# 운영체제 확인
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! command -v brew &> /dev/null; then
        echo "Homebrew 설치 중..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    echo "필요한 시스템 패키지 설치 중 (ffmpeg, git, cmake)..."
    brew install ffmpeg git cmake
    
    # Xcode 명령줄 도구 확인
    if ! xcode-select -p &> /dev/null; then
        echo "Xcode 명령줄 도구 설치 중..."
        xcode-select --install
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux (Ubuntu/Debian 기준)
    echo "필요한 시스템 패키지 설치 중..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg git cmake build-essential
else
    # Windows 또는 기타 OS
    echo "경고: 현재 운영체제에 대한 자동 설치 스크립트가 준비되어 있지 않습니다."
    echo "ffmpeg, git, cmake를 수동으로 설치해주세요."
fi

echo "========== Python 패키지 설치 =========="
# 패키지 설치 확인 함수
install_package() {
    echo "패키지 설치: $1"
    pip install "$1" || echo "경고: $1 설치 실패, 계속 진행합니다."
}

# 기본 패키지
install_package "requests"
install_package "aiohttp" 
install_package "aiofiles"

# PyTorch 설치 (macOS M1/M2의 경우 특별한 설치 방법 사용)
if [[ $(uname -s) == "Darwin" && $(uname -m) == "arm64" ]]; then
    echo "M1/M2 Mac용 PyTorch 설치 중..."
    install_package "torch"
    install_package "torchaudio"
    # CoreML 지원 패키지 설치
    install_package "coremltools"
    install_package "ane_transformers"
    install_package "openai-whisper"
else
    # 다른 시스템용 PyTorch 설치
    install_package "torch"
    install_package "torchaudio"
fi

echo "========== whisper.cpp 다운로드 및 빌드 =========="
if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggml-org/whisper.cpp.git
    cd whisper.cpp
    
    # 필요한 디렉토리 생성
    mkdir -p models
    
    # resources 폴더에서 모델 및 엔코더 파일 복사 (이미 존재한다고 가정)
    if [ -d "../resources" ]; then
        echo "resources 폴더에서 모델 파일 복사 중..."
        if [ -f "../resources/ggml-large-v3-turbo.bin" ]; then
            cp "../resources/ggml-large-v3-turbo.bin" models/
        fi
        
        if [ -d "../resources/ggml-large-v3-turbo-encoder.mlmodelc" ]; then
            cp -r "../resources/ggml-large-v3-turbo-encoder.mlmodelc" models/
        fi
    else
        echo "resources 폴더가 없습니다. 필요한 모델 파일이 있는지 확인하세요."
    fi
    
    # M1/M2 Mac인 경우 CoreML 지원 활성화
    if [[ $(uname -s) == "Darwin" && $(uname -m) == "arm64" ]]; then
        cmake -B build -DWHISPER_COREML=1
    else
        cmake -B build
    fi
    
    cmake --build build --config Release -j
    cd ..
else
    echo "whisper.cpp가 이미 존재합니다."
fi

echo "========== Silero VAD 모델 사전 다운로드 =========="
python -c "import torch; torch.hub.load('snakers4/silero-vad', model='silero_vad')" || echo "Silero VAD 모델 다운로드 실패, 나중에 재시도됩니다."

echo "========== 메인 스크립트 실행 =========="
python STT_Voice_Spliter.py