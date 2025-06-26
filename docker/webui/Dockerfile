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
COPY templates/ /app/templates/ 2>/dev/null || true
COPY static/ /app/static/ 2>/dev/null || true
WORKDIR /app

# 포트 설정
EXPOSE 7860

# 실행 명령
CMD ["python", "webui_service.py"]