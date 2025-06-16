# setup_and_run.ps1
# PowerShell 7.x 권장. 관리자 권한으로 실행 필요.

Write-Host "========== Miniconda/Anaconda 확인 =========="
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Miniconda/Anaconda가 설치되어 있지 않습니다. 먼저 설치해주세요."
    Write-Host "https://docs.conda.io/en/latest/miniconda.html 참고"
    exit 1
}
else {
    Write-Host "Miniconda/Anaconda가 이미 설치되어 있습니다."
}

Write-Host "========== Conda 환경 관리 =========="
conda deactivate
# 기존 환경 제거
conda env remove -n stt_env -y
Remove-Item -Recurse -Force "$env:CONDA_PREFIX\envs\stt_env" -ErrorAction SilentlyContinue

# 환경 생성
conda create -y -n stt_env python=3.11 tk ffmpeg

# 환경 활성화
conda activate stt_env

Write-Host "========== 시스템 종속성 설치 =========="
# Chocolatey가 없을 경우 안내
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey가 설치되어 있지 않으면 아래 명령어로 설치 후 재실행하세요:"
    Write-Host 'Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString("https://community.chocolatey.org/install.ps1"))'
    exit 1
}

# 필수 패키지 설치
choco install ffmpeg git cmake -y

Write-Host "========== Python 패키지 설치 =========="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install requests aiohttp aiofiles soundfile numpy

Write-Host "========== whisper.cpp 다운로드 및 빌드 =========="
if (-not (Test-Path ".\whisper.cpp")) {
    git clone https://github.com/ggml-org/whisper.cpp.git
    Set-Location whisper.cpp

    # 모델 파일 복사
    if (Test-Path "..\resources\ggml-large-v3-turbo.bin") {
        Copy-Item ..\resources\ggml-large-v3-turbo.bin .\models\
    }
    # Windows에서는 CoreML 관련 스킵
    cmake -B build -DGGML_CUDA=1
    cmake --build build -j --config Release
    Set-Location ..
}
else {
    Write-Host "whisper.cpp 폴더가 이미 존재합니다."
}

Write-Host "========== Silero VAD 모델 사전 다운로드 =========="
python -c "import torch; torch.hub.load('snakers4/silero-vad', model='silero_vad')"

Write-Host "========== 메인 스크립트 실행 =========="
python STT_Voice_Spliter.py

Pause