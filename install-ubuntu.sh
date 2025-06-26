#!/bin/bash

# DeepVoice STT Voice Splitter - Ubuntu 완전 설치 스크립트
# GPU 지원 Docker Compose 환경 자동 설치

set -e  # 에러 발생시 스크립트 중단

echo "🐧 DeepVoice STT Voice Splitter - Ubuntu 완전 설치 시작"
echo "============================================================="

# 색상 코드 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 시스템 정보 확인
print_status "시스템 정보 확인 중..."
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "Architecture: $(dpkg --print-architecture)"

# GPU 확인
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU 감지됨"
    nvidia-smi --query-gpu=name --format=csv,noheader
    GPU_SUPPORT=true
else
    print_warning "NVIDIA GPU가 감지되지 않았습니다. CPU 모드로 진행합니다."
    GPU_SUPPORT=false
fi

# 1. 시스템 업데이트
print_status "시스템 패키지 업데이트 중..."
sudo apt update && sudo apt upgrade -y

# 2. 필수 패키지 설치
print_status "필수 패키지 설치 중..."
sudo apt install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    wget \
    git \
    htop \
    tree

# 3. Docker 설치
print_status "Docker 설치 중..."

# 기존 Docker 제거 (있다면)
sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Docker GPG 키 추가
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Docker 저장소 추가
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 설치
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Docker 서비스 시작
sudo systemctl start docker
sudo systemctl enable docker

print_success "Docker 설치 완료"

# 4. 사용자 권한 설정
print_status "Docker 사용자 권한 설정 중..."
sudo usermod -aG docker $USER

print_success "사용자를 docker 그룹에 추가했습니다"
print_warning "변경사항 적용을 위해 재로그인하거나 'newgrp docker' 명령을 실행하세요"

# 5. NVIDIA Docker Runtime 설치 (GPU가 있는 경우)
if [ "$GPU_SUPPORT" = true ]; then
    print_status "NVIDIA Docker Runtime 설치 중..."
    
    # NVIDIA Docker 저장소 추가
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
       && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
       && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    # NVIDIA Docker 설치
    sudo apt update
    sudo apt install -y nvidia-docker2
    
    # Docker 재시작
    sudo systemctl restart docker
    
    print_success "NVIDIA Docker Runtime 설치 완료"
    
    # GPU Docker 테스트
    print_status "GPU Docker 테스트 중..."
    if docker run --rm --gpus all nvidia/cuda:11.8-devel-ubuntu22.04 nvidia-smi; then
        print_success "GPU Docker 테스트 성공"
    else
        print_error "GPU Docker 테스트 실패. 수동으로 확인이 필요합니다."
    fi
fi

# 6. Docker 설치 확인
print_status "Docker 설치 확인 중..."
docker --version
docker compose version

# 테스트 컨테이너 실행
if docker run --rm hello-world > /dev/null 2>&1; then
    print_success "Docker 설치 및 테스트 성공"
else
    print_error "Docker 테스트 실패. 'newgrp docker' 실행 후 다시 시도하세요."
fi

# 7. 프로젝트 설정
print_status "프로젝트 디렉토리 설정 중..."

# 현재 디렉토리가 프로젝트 루트인지 확인
if [ ! -f "docker-compose.yml" ] && [ ! -d "docker" ]; then
    print_error "프로젝트 루트 디렉토리에서 실행해주세요"
    exit 1
fi

# 브랜치 확인 및 전환
print_status "Git 브랜치 확인 중..."
if git branch --show-current | grep -q "docker-compose-migration"; then
    print_success "이미 docker-compose-migration 브랜치입니다"
else
    print_status "docker-compose-migration 브랜치로 전환 중..."
    git checkout docker-compose-migration
fi

# 필수 디렉토리 생성
print_status "필수 디렉토리 생성 중..."
mkdir -p {input,output,temp,config}
chmod 755 {input,output,temp,config}

print_success "디렉토리 생성 완료"

# 8. 모델 파일 확인
print_status "모델 파일 확인 중..."

echo "=== 모델 파일 상태 ==="

# Whisper 모델 확인
if [ -f "resources/ggml-large-v3-turbo.bin" ]; then
    print_success "Whisper 모델: 있음 ($(du -h resources/ggml-large-v3-turbo.bin | cut -f1))"
else
    print_warning "Whisper 모델: 없음"
    echo "다운로드 명령어:"
    echo "mkdir -p resources"
    echo "wget -O resources/ggml-large-v3-turbo.bin \\"
    echo "  https://huggingface.co/ggml-org/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
fi

# CosyVoice 모델 확인
if [ -d "CosyVoice/pretrained_models/CosyVoice2-0.5B" ] && [ "$(ls -A CosyVoice/pretrained_models/CosyVoice2-0.5B 2>/dev/null)" ]; then
    print_success "CosyVoice 모델: 있음"
else
    print_warning "CosyVoice 모델: 없음 또는 불완전"
    echo "CosyVoice 모델이 필요합니다. 별도로 다운로드하세요."
fi

# Gemma 모델 확인
if [ -f "gemma/gemma-3-12b-it-q4_0.gguf" ] || [ -f "gemma/gemma-3-27b-it-q4_0.gguf" ]; then
    print_success "Gemma 모델: 있음 (번역 기능 사용 가능)"
else
    print_warning "Gemma 모델: 없음 (번역 기능 비활성화)"
    echo "번역 기능을 사용하려면 Gemma 모델을 다운로드하세요."
fi

echo "========================"

# 9. 실행 스크립트 생성
print_status "실행 스크립트 생성 중..."

cat > run-deepvoice.sh << 'SCRIPT_EOF'
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
SCRIPT_EOF

chmod +x run-deepvoice.sh

print_success "실행 스크립트 생성 완료: ./run-deepvoice.sh"

# 10. 추가 유용한 스크립트 생성
cat > check-status.sh << 'SCRIPT_EOF'
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
SCRIPT_EOF

chmod +x check-status.sh

print_success "상태 확인 스크립트 생성 완료: ./check-status.sh"

# 11. 설치 완료 안내
echo ""
echo "🎉 설치 완료!"
echo "============="
echo ""
print_success "Docker 및 DeepVoice 환경 설치가 완료되었습니다!"
echo ""
echo "📋 다음 단계:"
echo "1. 터미널을 재시작하거나 'newgrp docker' 실행"
echo "2. 필요한 모델 파일 다운로드 (위의 안내 참조)"
echo "3. './run-deepvoice.sh' 실행으로 시스템 시작"
echo ""
echo "🔧 유용한 명령어:"
echo "• 시스템 시작: ./run-deepvoice.sh"
echo "• 상태 확인: ./check-status.sh"
echo "• 웹 UI 접속: http://localhost:7860"
echo "• 로그 확인: cd docker && docker compose logs -f"
echo "• 시스템 중지: cd docker && docker compose down"
echo ""
echo "📚 자세한 가이드: DOCKER-SETUP.md 파일 참조"
echo ""

if [ "$GPU_SUPPORT" = true ]; then
    print_success "GPU 지원이 활성화되었습니다!"
else
    print_warning "GPU 지원이 없습니다. CPU 모드로 실행됩니다."
fi

echo ""
print_warning "변경사항 적용을 위해 다음 중 하나를 실행하세요:"
echo "• newgrp docker  (현재 세션에서만 적용)"
echo "• 터미널 재시작 또는 시스템 재부팅"
