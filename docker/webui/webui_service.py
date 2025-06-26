#!/usr/bin/env python3
"""
DeepVoice Web UI 마이크로서비스
사용자 인터페이스를 제공하는 독립적인 서비스
"""

import os
import logging
import requests
import gradio as gr
from typing import Dict, List, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수
ORCHESTRATOR_URL = os.getenv('ORCHESTRATOR_URL', 'http://pipeline-orchestrator:8000')
ENABLE_REALTIME_LOG = os.getenv('ENABLE_REALTIME_LOG', 'true').lower() == 'true'
ENABLE_LIPSYNC_UI = os.getenv('ENABLE_LIPSYNC_UI', 'false').lower() == 'true'


def check_system_health():
    """시스템 전체 상태 확인"""
    try:
        response = requests.get(f"{ORCHESTRATOR_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": "Health check failed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def start_pipeline(
        input_file,
        target_languages,
        enable_lipsync,
        enable_speaker_splitting,
        translation_quality
):
    """파이프라인 처리 시작"""
    try:
        if input_file is None:
            return "❌ 파일을 선택해주세요", ""

        # 파일을 input 디렉토리로 복사 (실제로는 업로드 처리)
        file_path = f"/app/input/{input_file.name}"

        # 요청 데이터 구성
        request_data = {
            "input_file": file_path,
            "mode": "video_complete" if input_file.name.endswith(('.mp4', '.avi', '.mov')) else "audio_only",
            "target_languages": target_languages,
            "enable_lipsync": enable_lipsync and ENABLE_LIPSYNC_UI,
            "enable_speaker_splitting": enable_speaker_splitting,
            "translation_settings": {
                "quality_mode": translation_quality
            }
        }

        # 파이프라인 시작 요청
        response = requests.post(
            f"{ORCHESTRATOR_URL}/pipeline/process",
            json=request_data,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            job_id = result.get("job_id", "")
            message = result.get("message", "처리 시작됨")

            return f"✅ {message}", job_id
        else:
            return f"❌ 처리 시작 실패: {response.text}", ""

    except Exception as e:
        logger.error(f"Pipeline start error: {e}")
        return f"❌ 오류 발생: {str(e)}", ""


def check_pipeline_status(job_id):
    """파이프라인 처리 상태 확인"""
    try:
        if not job_id:
            return "❌ Job ID가 없습니다", 0, {}

        response = requests.get(f"{ORCHESTRATOR_URL}/pipeline/status/{job_id}", timeout=5)

        if response.status_code == 200:
            result = response.json()
            status = result.get("status", "unknown")
            progress = result.get("progress", 0)
            message = result.get("message", "")
            outputs = result.get("outputs", {})

            status_message = f"📊 상태: {status}\n📈 진행률: {progress}%\n💬 메시지: {message}"

            return status_message, progress, outputs
        else:
            return f"❌ 상태 확인 실패: {response.text}", 0, {}

    except Exception as e:
        logger.error(f"Status check error: {e}")
        return f"❌ 상태 확인 오류: {str(e)}", 0, {}


def create_gradio_interface():
    """Gradio 인터페이스 생성"""

    with gr.Blocks(title="🎙️ DeepVoice STT Voice Splitter", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎙️ DeepVoice STT Voice Splitter
        
        **AI 기반 음성 처리 및 다국어 번역 시스템**
        
        음성/영상 파일을 업로드하여 자동으로 텍스트 변환, 번역, 음성 합성을 수행합니다.
        """)

        # 시스템 상태 섹션
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🏥 시스템 상태")
                health_display = gr.JSON(label="서비스 상태", value=check_system_health())
                health_refresh_btn = gr.Button("🔄 상태 새로고침")
                health_refresh_btn.click(
                    fn=check_system_health,
                    outputs=health_display
                )

        gr.Markdown("---")

        # 파일 처리 섹션
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 📁 파일 업로드 및 설정")

                input_file = gr.File(
                    label="🎵 음성/영상 파일 선택",
                    file_types=[".wav", ".mp3", ".mp4", ".avi", ".mov"],
                    type="filepath"
                )

                with gr.Row():
                    target_languages = gr.CheckboxGroup(
                        choices=["english", "japanese", "chinese"],
                        value=["english"],
                        label="🌐 번역 언어 선택"
                    )

                    translation_quality = gr.Radio(
                        choices=["fast", "balanced", "high"],
                        value="balanced",
                        label="🎯 번역 품질"
                    )

                with gr.Row():
                    enable_speaker_splitting = gr.Checkbox(
                        label="🗣️ 화자 분리 활성화",
                        value=True
                    )

                    if ENABLE_LIPSYNC_UI:
                        enable_lipsync = gr.Checkbox(
                            label="💋 립싱크 활성화",
                            value=False
                        )
                    else:
                        enable_lipsync = gr.Checkbox(
                            label="💋 립싱크 활성화 (비활성화됨)",
                            value=False,
                            interactive=False
                        )

                process_btn = gr.Button("🚀 처리 시작", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("## 📊 처리 상태")

                job_id_display = gr.Textbox(
                    label="Job ID",
                    interactive=False,
                    placeholder="처리 시작 후 Job ID가 표시됩니다"
                )

                status_display = gr.Textbox(
                    label="처리 상태",
                    lines=5,
                    interactive=False,
                    placeholder="처리 상태가 여기에 표시됩니다"
                )

                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="진행률 (%)",
                    interactive=False
                )

                status_refresh_btn = gr.Button("🔄 상태 새로고침")

        gr.Markdown("---")

        # 결과 섹션
        with gr.Row():
            gr.Markdown("## 📥 처리 결과")
            results_display = gr.JSON(label="출력 파일 목록")

        # 이벤트 핸들러
        process_btn.click(
            fn=start_pipeline,
            inputs=[
                input_file,
                target_languages,
                enable_lipsync,
                enable_speaker_splitting,
                translation_quality
            ],
            outputs=[status_display, job_id_display]
        )

        status_refresh_btn.click(
            fn=check_pipeline_status,
            inputs=[job_id_display],
            outputs=[status_display, progress_bar, results_display]
        )

        # 자동 새로고침 (5초마다)
        if ENABLE_REALTIME_LOG:
            demo.load(
                fn=check_pipeline_status,
                inputs=[job_id_display],
                outputs=[status_display, progress_bar, results_display],
                every=5
            )

        gr.Markdown("""
        ---
        ### 📚 사용 가이드
        
        1. **파일 업로드**: 처리할 음성 또는 영상 파일을 선택합니다
        2. **언어 설정**: 번역할 언어를 선택합니다
        3. **옵션 설정**: 화자 분리, 립싱크 등의 옵션을 설정합니다
        4. **처리 시작**: '처리 시작' 버튼을 클릭합니다
        5. **상태 확인**: 진행률과 상태를 확인하며 완료를 기다립니다
        6. **결과 다운로드**: 처리 완료 후 결과 파일을 다운로드합니다
        
        ### ⚙️ 지원 형식
        - **음성**: WAV, MP3
        - **영상**: MP4, AVI, MOV
        
        ### 🔧 기술 스택
        - **STT**: Whisper Large v3 Turbo
        - **번역**: Gemma 3 LLM
        - **TTS**: CosyVoice2
        - **립싱크**: LatentSync (선택사항)
        """)

    return demo


def main():
    """메인 함수"""
    logger.info("Starting DeepVoice Web UI Service...")

    # Gradio 인터페이스 생성
    demo = create_gradio_interface()

    # 서버 시작
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()
