FROM python:3.10-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 애플리케이션 코드 복사
COPY *.py /app/
WORKDIR /app

# 디렉토리 생성
RUN mkdir -p /app/{input,output,temp,config}

# 포트 설정
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 실행 명령
CMD ["python", "orchestrator_service.py"]