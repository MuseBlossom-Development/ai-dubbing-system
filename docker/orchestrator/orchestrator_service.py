#!/usr/bin/env python3
"""
DeepVoice 파이프라인 오케스트레이터
각 마이크로서비스를 조율하여 전체 음성 처리 파이프라인을 실행
"""

import os
import json
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from enum import Enum

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DeepVoice Pipeline Orchestrator", version="1.0.0")

# 서비스 URL 환경 변수
WHISPER_SERVICE_URL = os.getenv('WHISPER_SERVICE_URL', 'http://whisper-stt:8000')
TRANSLATOR_SERVICE_URL = os.getenv('TRANSLATOR_SERVICE_URL', 'http://gemma-translator:8000')
TTS_SERVICE_URL = os.getenv('TTS_SERVICE_URL', 'http://cosyvoice-tts:8000')
LIPSYNC_SERVICE_URL = os.getenv('LIPSYNC_SERVICE_URL', 'http://latentsync-lipsync:8000')
AUDIO_PROCESSOR_URL = os.getenv('AUDIO_PROCESSOR_URL', 'http://audio-processor:8000')

# 설정
ENABLE_LIPSYNC = os.getenv('ENABLE_LIPSYNC', 'true').lower() == 'true'
PIPELINE_MODE = os.getenv('PIPELINE_MODE', 'complete')


class PipelineStatus(str, Enum):
    """파이프라인 상태"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingMode(str, Enum):
    """처리 모드"""
    AUDIO_ONLY = "audio_only"
    VIDEO_COMPLETE = "video_complete"
    CUSTOM = "custom"


class PipelineRequest(BaseModel):
    """파이프라인 요청 모델"""
    input_file: str
    mode: ProcessingMode = ProcessingMode.VIDEO_COMPLETE
    target_languages: List[str] = ["english"]
    enable_lipsync: bool = True
    enable_speaker_splitting: bool = True
    translation_settings: Dict[str, Any] = {}
    tts_settings: Dict[str, Any] = {}
    output_settings: Dict[str, Any] = {}


class PipelineResponse(BaseModel):
    """파이프라인 응답 모델"""
    job_id: str
    status: PipelineStatus
    message: str
    progress: float = 0.0
    outputs: Dict[str, str] = {}
    error: Optional[str] = None


class ServiceHealthCheck(BaseModel):
    """서비스 헬스체크 모델"""
    whisper_stt: bool = False
    translator: bool = False
    tts: bool = False
    lipsync: bool = False
    audio_processor: bool = False


# 전역 작업 저장소 (실제 환경에서는 Redis 등 사용)
jobs_storage: Dict[str, Dict] = {}


@app.get("/health")
async def health_check():
    """전체 시스템 헬스체크"""
    try:
        health_status = ServiceHealthCheck()

        # 각 서비스 헬스체크
        async with aiohttp.ClientSession() as session:
            # Whisper STT 서비스
            try:
                async with session.get(f"{WHISPER_SERVICE_URL}/health", timeout=5) as resp:
                    health_status.whisper_stt = resp.status == 200
            except:
                pass

            # 번역 서비스
            try:
                async with session.get(f"{TRANSLATOR_SERVICE_URL}/health", timeout=5) as resp:
                    health_status.translator = resp.status == 200
            except:
                pass

            # TTS 서비스
            try:
                async with session.get(f"{TTS_SERVICE_URL}/health", timeout=5) as resp:
                    health_status.tts = resp.status == 200
            except:
                pass

            # 립싱크 서비스 (선택적)
            if ENABLE_LIPSYNC:
                try:
                    async with session.get(f"{LIPSYNC_SERVICE_URL}/health", timeout=5) as resp:
                        health_status.lipsync = resp.status == 200
                except:
                    pass

            # 오디오 프로세서
            try:
                async with session.get(f"{AUDIO_PROCESSOR_URL}/health", timeout=5) as resp:
                    health_status.audio_processor = resp.status == 200
            except:
                pass

        return {
            "status": "healthy",
            "services": health_status.dict(),
            "pipeline_mode": PIPELINE_MODE,
            "lipsync_enabled": ENABLE_LIPSYNC
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/process", response_model=PipelineResponse)
async def start_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """파이프라인 처리 시작"""
    try:
        # 작업 ID 생성
        import uuid
        job_id = str(uuid.uuid4())

        # 작업 정보 저장
        jobs_storage[job_id] = {
            "status": PipelineStatus.PENDING,
            "request": request.dict(),
            "progress": 0.0,
            "outputs": {},
            "created_at": asyncio.get_event_loop().time()
        }

        # 백그라운드에서 파이프라인 실행
        background_tasks.add_task(execute_pipeline, job_id, request)

        return PipelineResponse(
            job_id=job_id,
            status=PipelineStatus.PENDING,
            message="Pipeline started successfully"
        )

    except Exception as e:
        logger.error(f"Pipeline start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/status/{job_id}", response_model=PipelineResponse)
async def get_pipeline_status(job_id: str):
    """파이프라인 상태 조회"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = jobs_storage[job_id]

    return PipelineResponse(
        job_id=job_id,
        status=job_data["status"],
        message=job_data.get("message", ""),
        progress=job_data.get("progress", 0.0),
        outputs=job_data.get("outputs", {}),
        error=job_data.get("error")
    )


@app.delete("/pipeline/{job_id}")
async def cancel_pipeline(job_id: str):
    """파이프라인 취소"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")

    jobs_storage[job_id]["status"] = PipelineStatus.FAILED
    jobs_storage[job_id]["message"] = "Cancelled by user"

    return {"message": "Pipeline cancelled successfully"}


async def execute_pipeline(job_id: str, request: PipelineRequest):
    """파이프라인 실행 (백그라운드 작업)"""
    try:
        # 작업 상태 업데이트
        jobs_storage[job_id]["status"] = PipelineStatus.PROCESSING
        jobs_storage[job_id]["message"] = "Starting pipeline execution"

        logger.info(f"Starting pipeline execution for job {job_id}")

        # Step 1: 오디오/비디오 전처리
        await update_progress(job_id, 10, "Processing input file")
        processed_audio_path = await process_input_file(request.input_file)

        # Step 2: STT (Whisper)
        await update_progress(job_id, 25, "Speech-to-text processing")
        transcription_result = await call_whisper_service(processed_audio_path)

        # Step 3: 번역 (각 언어별)
        await update_progress(job_id, 40, "Translation processing")
        translations = {}
        for lang in request.target_languages:
            translation_result = await call_translation_service(
                transcription_result["text"],
                lang,
                request.translation_settings
            )
            translations[lang] = translation_result

        # Step 4: TTS (음성 합성)
        await update_progress(job_id, 60, "Text-to-speech processing")
        synthesized_audios = {}
        for lang, translation in translations.items():
            tts_result = await call_tts_service(
                translation["text"],
                lang,
                processed_audio_path,
                request.tts_settings
            )
            synthesized_audios[lang] = tts_result

        # Step 5: 오디오 병합 및 동기화
        await update_progress(job_id, 75, "Audio merging and synchronization")
        merged_audios = {}
        for lang, audio_data in synthesized_audios.items():
            merged_audio = await merge_audio_segments(
                audio_data,
                transcription_result["segments"]
            )
            merged_audios[lang] = merged_audio

        # Step 6: 립싱크 (비디오 모드이고 활성화된 경우)
        final_outputs = {}
        if request.mode == ProcessingMode.VIDEO_COMPLETE and request.enable_lipsync:
            await update_progress(job_id, 90, "Lip-sync processing")
            for lang, audio_path in merged_audios.items():
                lipsync_result = await call_lipsync_service(
                    request.input_file,
                    audio_path
                )
                final_outputs[f"{lang}_video"] = lipsync_result["output_path"]
        else:
            # 오디오만 출력
            final_outputs = merged_audios

        # 완료
        await update_progress(job_id, 100, "Pipeline completed successfully")
        jobs_storage[job_id]["status"] = PipelineStatus.COMPLETED
        jobs_storage[job_id]["outputs"] = final_outputs

        logger.info(f"Pipeline execution completed for job {job_id}")

    except Exception as e:
        logger.error(f"Pipeline execution failed for job {job_id}: {e}")
        jobs_storage[job_id]["status"] = PipelineStatus.FAILED
        jobs_storage[job_id]["error"] = str(e)
        jobs_storage[job_id]["message"] = f"Pipeline failed: {str(e)}"


async def update_progress(job_id: str, progress: float, message: str):
    """진행률 업데이트"""
    if job_id in jobs_storage:
        jobs_storage[job_id]["progress"] = progress
        jobs_storage[job_id]["message"] = message
        logger.info(f"Job {job_id}: {progress}% - {message}")


async def process_input_file(input_path: str) -> str:
    """입력 파일 전처리"""
    try:
        async with aiohttp.ClientSession() as session:
            data = {"input_path": input_path}
            async with session.post(f"{AUDIO_PROCESSOR_URL}/process", json=data) as resp:
                if resp.status != 200:
                    raise Exception(f"Audio processing failed: {await resp.text()}")
                result = await resp.json()
                return result["processed_audio_path"]
    except Exception as e:
        logger.error(f"Input processing failed: {e}")
        raise


async def call_whisper_service(audio_path: str) -> Dict:
    """Whisper STT 서비스 호출"""
    try:
        async with aiohttp.ClientSession() as session:
            data = {"audio_path": audio_path}
            async with session.post(f"{WHISPER_SERVICE_URL}/transcribe", json=data) as resp:
                if resp.status != 200:
                    raise Exception(f"Whisper service failed: {await resp.text()}")
                return await resp.json()
    except Exception as e:
        logger.error(f"Whisper service call failed: {e}")
        raise


async def call_translation_service(text: str, target_lang: str, settings: Dict) -> Dict:
    """번역 서비스 호출"""
    try:
        async with aiohttp.ClientSession() as session:
            data = {
                "text": text,
                "target_language": target_lang,
                "settings": settings
            }
            async with session.post(f"{TRANSLATOR_SERVICE_URL}/translate", json=data) as resp:
                if resp.status != 200:
                    raise Exception(f"Translation service failed: {await resp.text()}")
                return await resp.json()
    except Exception as e:
        logger.error(f"Translation service call failed: {e}")
        raise


async def call_tts_service(text: str, lang: str, reference_audio: str, settings: Dict) -> Dict:
    """TTS 서비스 호출"""
    try:
        async with aiohttp.ClientSession() as session:
            data = {
                "text": text,
                "language": lang,
                "reference_audio": reference_audio,
                "settings": settings
            }
            async with session.post(f"{TTS_SERVICE_URL}/synthesize", json=data) as resp:
                if resp.status != 200:
                    raise Exception(f"TTS service failed: {await resp.text()}")
                return await resp.json()
    except Exception as e:
        logger.error(f"TTS service call failed: {e}")
        raise


async def call_lipsync_service(video_path: str, audio_path: str) -> Dict:
    """립싱크 서비스 호출"""
    try:
        async with aiohttp.ClientSession() as session:
            data = {
                "video_path": video_path,
                "audio_path": audio_path
            }
            async with session.post(f"{LIPSYNC_SERVICE_URL}/process", json=data) as resp:
                if resp.status != 200:
                    raise Exception(f"Lipsync service failed: {await resp.text()}")
                return await resp.json()
    except Exception as e:
        logger.error(f"Lipsync service call failed: {e}")
        raise


async def merge_audio_segments(audio_data: Dict, segments: List[Dict]) -> str:
    """오디오 세그먼트 병합"""
    # 이 부분은 실제 구현에서는 오디오 처리 서비스로 위임
    try:
        async with aiohttp.ClientSession() as session:
            data = {
                "audio_data": audio_data,
                "segments": segments
            }
            async with session.post(f"{AUDIO_PROCESSOR_URL}/merge", json=data) as resp:
                if resp.status != 200:
                    raise Exception(f"Audio merge failed: {await resp.text()}")
                result = await resp.json()
                return result["merged_path"]
    except Exception as e:
        logger.error(f"Audio merge failed: {e}")
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
