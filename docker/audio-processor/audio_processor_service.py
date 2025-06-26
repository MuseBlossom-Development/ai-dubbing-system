#!/usr/bin/env python3
"""
오디오/비디오 처리 마이크로서비스
파일 변환, 분할, 병합 등을 담당하는 독립적인 서비스
"""

import os
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from pydub import AudioSegment

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepVoice Audio Processor Service", version="1.0.0")

# 환경 변수
FFMPEG_THREADS = int(os.getenv('FFMPEG_THREADS', '4'))
ENABLE_GPU_ACCELERATION = os.getenv('ENABLE_GPU_ACCELERATION', 'true').lower() == 'true'


class ProcessRequest(BaseModel):
    """처리 요청 모델"""
    input_path: str
    output_format: str = "wav"
    sample_rate: int = 16000
    channels: int = 1


class ProcessResponse(BaseModel):
    """처리 응답 모델"""
    processed_audio_path: str
    original_path: str
    format: str
    duration: float
    sample_rate: int


class MergeRequest(BaseModel):
    """병합 요청 모델"""
    audio_data: Dict
    segments: List[Dict]


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str
    ffmpeg_available: bool
    gpu_acceleration: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서비스 상태 확인"""
    # FFmpeg 사용 가능 여부 확인
    ffmpeg_available = True
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except:
        ffmpeg_available = False

    return HealthResponse(
        status="healthy" if ffmpeg_available else "ffmpeg_missing",
        ffmpeg_available=ffmpeg_available,
        gpu_acceleration=ENABLE_GPU_ACCELERATION
    )


@app.post("/process", response_model=ProcessResponse)
async def process_audio(request: ProcessRequest):
    """오디오/비디오 파일 처리"""
    try:
        input_path = request.input_path

        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail="Input file not found")

        # 파일 확장자 확인
        input_ext = Path(input_path).suffix.lower()
        is_video = input_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']

        # 출력 파일 경로 생성
        output_dir = os.path.dirname(input_path).replace('/input', '/output')
        os.makedirs(output_dir, exist_ok=True)

        base_name = Path(input_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_processed.{request.output_format}")

        if is_video:
            # 비디오에서 오디오 추출
            logger.info(f"Extracting audio from video: {input_path}")
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vn',  # 비디오 스트림 제외
                '-acodec', 'pcm_s16le',
                '-ar', str(request.sample_rate),
                '-ac', str(request.channels),
                '-threads', str(FFMPEG_THREADS),
                '-y', output_path
            ]
        else:
            # 오디오 형식 변환
            logger.info(f"Converting audio: {input_path}")
            cmd = [
                'ffmpeg', '-i', input_path,
                '-acodec', 'pcm_s16le',
                '-ar', str(request.sample_rate),
                '-ac', str(request.channels),
                '-threads', str(FFMPEG_THREADS),
                '-y', output_path
            ]

        # FFmpeg 실행
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {result.stderr}")

        # 처리된 파일 정보 가져오기
        audio = AudioSegment.from_file(output_path)
        duration = len(audio) / 1000.0  # 초 단위

        return ProcessResponse(
            processed_audio_path=output_path,
            original_path=input_path,
            format=request.output_format,
            duration=duration,
            sample_rate=request.sample_rate
        )

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_vocals")
async def extract_vocals(input_path: str):
    """보컬과 배경음 분리"""
    try:
        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail="Input file not found")

        # 출력 디렉토리 설정
        output_dir = os.path.dirname(input_path).replace('/input', '/output')
        os.makedirs(output_dir, exist_ok=True)

        base_name = Path(input_path).stem
        vocals_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
        background_path = os.path.join(output_dir, f"{base_name}_background.wav")

        # 간단한 보컬 분리 (FFmpeg 사용)
        # 보컬 추출 (중앙 채널)
        vocals_cmd = [
            'ffmpeg', '-i', input_path,
            '-af', 'pan=mono|c0=0.5*c0+-0.5*c1',
            '-y', vocals_path
        ]

        # 배경음 추출 (사이드 채널)
        background_cmd = [
            'ffmpeg', '-i', input_path,
            '-af', 'pan=mono|c0=0.5*c0+0.5*c1',
            '-y', background_path
        ]

        # 보컬 추출 실행
        result = subprocess.run(vocals_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Vocal extraction failed: {result.stderr}")

        # 배경음 추출 실행
        result = subprocess.run(background_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Background extraction failed: {result.stderr}")

        return {
            "vocals_path": vocals_path,
            "background_path": background_path,
            "original_path": input_path
        }

    except Exception as e:
        logger.error(f"Vocal extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/merge")
async def merge_audio_segments(request: MergeRequest):
    """오디오 세그먼트 병합"""
    try:
        # 임시 출력 파일 생성
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            output_path = tmp_file.name

        # 세그먼트 정보로 병합 로직 구현
        logger.info(f"Merging {len(request.segments)} audio segments")

        # 병합된 오디오 생성 (placeholder)
        merged_audio = AudioSegment.empty()

        for segment in request.segments:
            # 각 세그먼트 처리 로직
            start_ms = segment.get('start', 0) * 1000
            end_ms = segment.get('end', 1) * 1000

            # 무음 추가 (실제로는 합성된 오디오)
            duration_ms = end_ms - start_ms
            silence = AudioSegment.silent(duration=duration_ms)
            merged_audio += silence

        # 병합된 오디오 저장
        merged_audio.export(output_path, format="wav")

        return {"merged_path": output_path}

    except Exception as e:
        logger.error(f"Audio merge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_process")
async def upload_and_process(file: UploadFile = File(...)):
    """파일 업로드 및 처리"""
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # 처리 요청
        request = ProcessRequest(input_path=tmp_path)
        result = await process_audio(request)

        # 임시 파일 삭제
        os.unlink(tmp_path)

        return result

    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
