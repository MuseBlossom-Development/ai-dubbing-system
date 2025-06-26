#!/usr/bin/env python3
"""
Whisper STT 마이크로서비스
음성 파일을 받아 텍스트로 변환하는 독립적인 서비스
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import asyncio
import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepVoice Whisper STT Service", version="1.0.0")

# 환경 변수
MODEL_SIZE = os.getenv('MODEL_SIZE', 'large-v3-turbo')
LANGUAGE = os.getenv('LANGUAGE', 'ko')
VAD_THRESHOLD = float(os.getenv('VAD_THRESHOLD', '0.6'))

# 경로 설정
WHISPER_BIN = "/app/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH = f"/app/models/ggml-{MODEL_SIZE}.bin"


class TranscriptionRequest(BaseModel):
    """전사 요청 모델"""
    audio_path: str
    language: Optional[str] = None
    vad_threshold: Optional[float] = None
    enable_timestamps: bool = True


class TranscriptionResponse(BaseModel):
    """전사 응답 모델"""
    text: str
    segments: List[Dict]
    duration: float
    language: str


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str
    model_loaded: bool
    gpu_available: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서비스 상태 확인"""
    try:
        # 모델 파일 존재 확인
        model_exists = os.path.exists(MODEL_PATH)

        # GPU 사용 가능 여부 확인
        gpu_available = False
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            gpu_available = result.returncode == 0
        except:
            pass

        return HealthResponse(
            status="healthy" if model_exists else "model_missing",
            model_loaded=model_exists,
            gpu_available=gpu_available
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            gpu_available=False
        )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """음성 파일 전사"""
    try:
        if not os.path.exists(request.audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found")

        # Whisper 명령어 구성
        language = request.language or LANGUAGE
        vad_threshold = request.vad_threshold or VAD_THRESHOLD

        cmd = [
            WHISPER_BIN,
            "-m", MODEL_PATH,
            "-f", request.audio_path,
            "-l", language,
            "--output-srt",
            "--output-txt",
            "--print-colors",
            f"--vad-thold", str(vad_threshold)
        ]

        # GPU 사용 가능 시 활성화
        if os.path.exists('/usr/bin/nvidia-smi'):
            cmd.append("--gpu")

        logger.info(f"Running Whisper with command: {' '.join(cmd)}")

        # Whisper 실행
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(f"Whisper failed: {stderr.decode()}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {stderr.decode()}")

        # 결과 파일 읽기
        audio_name = Path(request.audio_path).stem
        txt_file = f"{audio_name}.txt"
        srt_file = f"{audio_name}.srt"

        text = ""
        segments = []
        duration = 0.0

        # 텍스트 파일 읽기
        if os.path.exists(txt_file):
            async with aiofiles.open(txt_file, 'r', encoding='utf-8') as f:
                text = await f.read()

        # SRT 파일 파싱
        if os.path.exists(srt_file):
            segments, duration = await parse_srt_file(srt_file)

        return TranscriptionResponse(
            text=text.strip(),
            segments=segments,
            duration=duration,
            language=language
        )

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe_upload")
async def transcribe_upload(file: UploadFile = File(...)):
    """업로드된 파일 전사"""
    try:
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # 전사 요청
        request = TranscriptionRequest(audio_path=tmp_path)
        result = await transcribe_audio(request)

        # 임시 파일 삭제
        os.unlink(tmp_path)

        return result

    except Exception as e:
        logger.error(f"Upload transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def parse_srt_file(srt_path: str) -> tuple[List[Dict], float]:
    """SRT 파일 파싱"""
    segments = []
    max_end_time = 0.0

    try:
        async with aiofiles.open(srt_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        # SRT 형식 파싱
        blocks = content.strip().split('\n\n')

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # 시간 정보 파싱
                time_line = lines[1]
                if ' --> ' in time_line:
                    start_str, end_str = time_line.split(' --> ')
                    start_ms = parse_time_to_ms(start_str.strip())
                    end_ms = parse_time_to_ms(end_str.strip())

                    # 텍스트 추출
                    text = '\n'.join(lines[2:])

                    segments.append({
                        'start': start_ms / 1000.0,  # 초 단위로 변환
                        'end': end_ms / 1000.0,
                        'text': text.strip()
                    })

                    max_end_time = max(max_end_time, end_ms / 1000.0)

        return segments, max_end_time

    except Exception as e:
        logger.error(f"SRT parsing error: {e}")
        return [], 0.0


def parse_time_to_ms(time_str: str) -> int:
    """시간 문자열을 밀리초로 변환"""
    try:
        # 형식: HH:MM:SS,mmm
        time_part, ms_part = time_str.split(',')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)

        total_ms = (h * 3600 + m * 60 + s) * 1000 + ms
        return total_ms

    except:
        return 0


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
