FROM nvidia/cuda:11.8-devel-ubuntu22.04

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# whisper.cpp 빌드
WORKDIR /app
RUN git clone https://github.com/ggerganov/whisper.cpp.git
WORKDIR /app/whisper.cpp
RUN mkdir build && cd build && \
    cmake .. -DWHISPER_CUDA=ON && \
    cmake --build . --config Release

# 모델 디렉토리 생성
RUN mkdir -p /app/models

# 애플리케이션 코드 복사
COPY *.py /app/
COPY models/ /app/models/

# 작업 디렉토리 설정
WORKDIR /app

# 포트 설정
EXPOSE 8000

# 실행 명령
CMD ["python3", "whisper_service.py"]