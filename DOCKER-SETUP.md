# 🐳 Docker Compose 설정 가이드

## 📋 개요

DeepVoice STT Voice Splitter를 Docker Compose를 사용하여 마이크로서비스 아키텍처로 실행하는 가이드입니다.

## 🏗️ 아키텍처 구성

### 서비스 구성도

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                          │
│  (deepvoice-network: 172.20.0.0/16)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Web UI     │    │ Orchestrator│    │Audio Processor│   │
│  │   :7860     │◄───┤   :8000     │◄───┤   :8005     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                            │                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │Whisper STT  │    │  Gemma      │    │ CosyVoice   │     │
│  │   :8001     │◄───┤Translator   │◄───┤   TTS       │     │
│  │   (GPU)     │    │   :8002     │    │   :8003     │     │
│  └─────────────┘    └─────────────┘    │   (GPU)     │     │
│                                        └─────────────┘     │
│  ┌─────────────┐                                           │ 
│  │LatentSync   │                                           │
│  │ Lipsync     │ (선택사항)                                 │
│  │   :8004     │                                           │
│  │   (GPU)     │                                           │
│  └─────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 시작하기

### 1. 사전 요구사항

#### 시스템 요구사항

```bash
# 최소 요구사항
- CPU: 8 cores
- RAM: 16GB
- GPU: NVIDIA GTX 1060 6GB (선택사항)
- Storage: 100GB

# 권장 요구사항  
- CPU: 16 cores
- RAM: 32GB
- GPU: NVIDIA RTX 3080 10GB
- Storage: 200GB SSD
```

#### 필수 소프트웨어

```bash
# Docker & Docker Compose 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose V2 설치 (이미 포함됨)
docker compose version

# NVIDIA Docker Runtime (GPU 사용시)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. 프로젝트 설정

#### 브랜치 전환

```bash
git checkout docker-compose-migration
```

#### 디렉토리 구조 생성

```bash
# 필수 디렉토리 생성
mkdir -p {input,output,temp,config}

# 권한 설정
chmod 755 {input,output,temp,config}
```

#### 모델 파일 준비

```bash
# 1. Whisper 모델 (필수)
mkdir -p resources
wget -O resources/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggml-org/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

# 2. CosyVoice2 모델 (필수)
# CosyVoice/pretrained_models/ 디렉토리에 모델 파일 확인
ls -la CosyVoice/pretrained_models/CosyVoice2-0.5B/

# 3. Gemma3 모델 (선택사항)
# gemma/gemma-3-12b-it-q4_0.gguf 파일 확인
ls -la gemma/

# 4. LatentSync 모델 (립싱크 사용시)
# LatentSync/checkpoints/ 또는 LatentSync/checkpoints_v1.6/ 확인
ls -la LatentSync/checkpoints*/
```

### 3. 실행 방법

#### Option A: 립싱크 제외 버전 (권장)

```bash
cd docker
docker compose -f docker-compose.no-lipsync.yml up --build
```

#### Option B: 전체 버전 (립싱크 포함)

```bash  
cd docker
docker compose up --build
```

#### 백그라운드 실행

```bash
docker compose -f docker-compose.no-lipsync.yml up -d --build
```

### 4. 서비스 상태 확인

#### 헬스체크

```bash
# 전체 시스템 상태
curl http://localhost:8000/health

# 개별 서비스 상태
curl http://localhost:8001/health  # Whisper STT
curl http://localhost:8002/health  # Gemma Translator
curl http://localhost:8003/health  # CosyVoice TTS
curl http://localhost:8005/health  # Audio Processor
```

#### 컨테이너 상태

```bash
# 실행 중인 컨테이너 확인
docker compose ps

# 로그 확인
docker compose logs -f

# 특정 서비스 로그
docker compose logs -f whisper-stt
```

## 🔧 상세 설정

### 환경 변수 설정

#### docker-compose.yml 커스터마이징

```yaml
services:
  whisper-stt:
    environment:
      - MODEL_SIZE=large-v3-turbo    # 모델 크기 조정
      - LANGUAGE=ko                  # 기본 언어
      - VAD_THRESHOLD=0.6           # VAD 임계값
      
  gemma-translator:
    environment:
      - MODEL_PATH=/app/models/gemma-3-27b-it-q4_0.gguf  # 더 큰 모델 사용
      - MAX_TOKENS=8192             # 컨텍스트 증가
      - TEMPERATURE=0.3             # 번역 온도 조정
      
  cosyvoice-tts:
    environment:
      - SAMPLE_RATE=48000           # 고품질 샘플레이트
      - ENABLE_ZERO_SHOT=true       # Zero-shot 활성화
```

#### .env 파일 사용

```bash
# docker/.env 파일 생성
cat > docker/.env << EOF
# Whisper 설정
WHISPER_MODEL_SIZE=large-v3-turbo
WHISPER_LANGUAGE=ko
WHISPER_VAD_THRESHOLD=0.6

# Gemma 설정
GEMMA_MODEL_PATH=/app/models/gemma-3-12b-it-q4_0.gguf
GEMMA_MAX_TOKENS=4096
GEMMA_TEMPERATURE=0.2

# CosyVoice 설정
COSYVOICE_SAMPLE_RATE=24000
COSYVOICE_ENABLE_ZERO_SHOT=true

# 파이프라인 설정
ENABLE_LIPSYNC=false
PIPELINE_MODE=audio_complete
EOF
```

### 리소스 제한 설정

#### GPU 메모리 제한

```yaml
services:
  whisper-stt:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 8G
          
  cosyvoice-tts:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']  # 특정 GPU 지정
              capabilities: [gpu]
```

#### CPU/메모리 제한

```yaml
services:
  gemma-translator:
    deploy:
      resources:
        limits:
          cpus: '8.0'
          memory: 16G
        reservations:
          cpus: '4.0'
          memory: 8G
```

## 📊 모니터링 및 관리

### 실시간 모니터링

#### 리소스 사용량 확인

```bash
# 컨테이너별 리소스 사용량
docker stats

# GPU 사용량 모니터링  
watch -n 1 nvidia-smi

# 디스크 사용량
df -h
du -sh output/
```

#### 로그 분석

```bash
# 실시간 로그 스트림
docker compose logs -f --tail=100

# 특정 시간대 로그
docker compose logs --since="2024-01-01T10:00:00"

# 에러 로그만 필터링
docker compose logs 2>&1 | grep -i error
```

### 성능 튜닝

#### 병렬 처리 설정

```yaml
services:
  whisper-stt:
    deploy:
      replicas: 2  # 다중 인스턴스
    ports:
      - "8001-8002:8000"  # 포트 범위 설정
```

#### 캐시 최적화

```yaml
services:
  cosyvoice-tts:
    volumes:
      - model-cache:/root/.cache  # 모델 캐시 볼륨
    tmpfs:
      - /tmp:size=2G  # 임시 파일 메모리 저장
```

## 🔧 개발 및 디버깅

### 개발 모드 설정

#### docker-compose.override.yml 생성

```yaml
# 개발용 오버라이드 설정
version: '3.8'
services:
  whisper-stt:
    volumes:
      - ../docker/whisper/whisper_service.py:/app/whisper_service.py
    command: ["python", "-u", "whisper_service.py"]
    
  pipeline-orchestrator:
    volumes:
      - ../docker/orchestrator:/app
    environment:
      - DEBUG=True
```

#### 코드 변경 후 재시작

```bash
# 특정 서비스만 재빌드
docker compose build whisper-stt
docker compose up -d whisper-stt

# 전체 재빌드
docker compose down
docker compose up --build
```

### 트러블슈팅

#### 일반적인 문제 해결

1. **컨테이너 시작 실패**

```bash
# 컨테이너 상태 확인
docker compose ps -a

# 실패한 컨테이너 로그 확인
docker compose logs [service-name]

# 컨테이너 수동 실행으로 디버깅
docker run -it --rm deepvoice-whisper-stt /bin/bash
```

2. **모델 파일 로딩 실패**

```bash
# 볼륨 마운트 확인
docker compose exec whisper-stt ls -la /app/models/

# 파일 권한 확인
docker compose exec whisper-stt ls -la /app/models/ggml-large-v3-turbo.bin

# 수동으로 모델 파일 복사
docker cp resources/ggml-large-v3-turbo.bin container_name:/app/models/
```

3. **네트워크 연결 문제**

```bash
# 네트워크 상태 확인
docker network ls
docker network inspect docker_deepvoice-network

# 서비스간 연결 테스트
docker compose exec pipeline-orchestrator curl whisper-stt:8000/health
```

4. **GPU 인식 문제**

```bash
# NVIDIA Docker 런타임 확인
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# 컨테이너 내 GPU 확인
docker compose exec whisper-stt nvidia-smi
```

5. **메모리 부족**

```bash
# Docker 시스템 정리
docker system prune -a -f

# 미사용 볼륨 정리
docker volume prune -f

# 메모리 사용량 확인
docker compose exec whisper-stt free -h
```

## 🚀 프로덕션 배포

### 보안 설정

#### 방화벽 설정

```bash
# 필요한 포트만 열기
sudo ufw allow 8000  # Orchestrator
sudo ufw allow 7860  # Web UI

# 내부 서비스 포트는 차단
sudo ufw deny 8001,8002,8003,8004,8005
```

#### Docker Secrets 사용

```yaml
services:
  gemma-translator:
    secrets:
      - gemma_model_key
    environment:
      - MODEL_KEY_FILE=/run/secrets/gemma_model_key

secrets:
  gemma_model_key:
    file: ./secrets/gemma_key.txt
```

### 로드 밸런싱

#### Nginx 프록시 설정

```nginx
upstream deepvoice_orchestrator {
    server localhost:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://deepvoice_orchestrator;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 백업 및 복구

#### 데이터 백업

```bash
# 출력 데이터 백업
tar -czf backup_$(date +%Y%m%d).tar.gz output/

# 모델 파일 백업 (1회만)
tar -czf models_backup.tar.gz resources/ CosyVoice/ gemma/
```

#### 설정 백업

```bash
# Docker 설정 백업
cp -r docker/ docker_backup_$(date +%Y%m%d)/
```

## 📞 지원

문제가 발생하면 다음 정보와 함께 이슈를 생성해주세요:

```bash
# 시스템 정보 수집
echo "=== System Info ===" > debug_info.txt
uname -a >> debug_info.txt
docker --version >> debug_info.txt
docker compose version >> debug_info.txt

echo "=== GPU Info ===" >> debug_info.txt
nvidia-smi >> debug_info.txt 2>&1

echo "=== Container Status ===" >> debug_info.txt
docker compose ps >> debug_info.txt

echo "=== Service Logs ===" >> debug_info.txt
docker compose logs --tail=50 >> debug_info.txt
```