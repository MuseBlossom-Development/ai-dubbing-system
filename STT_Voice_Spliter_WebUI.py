import os
import sys
import gradio as gr
import threading
import json
import subprocess
import shutil
from pathlib import Path
from pydub import AudioSegment


# 환경 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 모듈화된 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CosyVoice'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CosyVoice', 'third_party'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CosyVoice', 'third_party', 'Matcha-TTS'))

from config import load_vad_config, save_vad_config
from main_processor import start_processing_with_settings
from whisper_processor import generate_srt_only
from audio_processor import parse_srt_segments, merge_segments_preserve_timing
from speaker_diarization import test_speaker_diarization
from utils import log_message


def apply_lip_sync(video_path, audio_path, output_path, progress_callback=None):
    """
    LatentSync를 사용하여 립싱크 적용
    
    Args:
        video_path: 원본 비디오 경로
        audio_path: 번역된 음성 경로  
        output_path: 립싱크 적용된 비디오 출력 경로
        progress_callback: 진행률 콜백 함수
    
    Returns:
        bool: 성공 여부
        str: 결과 메시지
    """
    try:
        from datetime import datetime

        if progress_callback:
            progress_callback(0.1, "LatentSync 준비 중...")

        # LatentSync 디렉토리 경로
        latentsync_dir = os.path.join(os.path.dirname(__file__), 'LatentSync')

        if not os.path.exists(latentsync_dir):
            return False, "LatentSync가 설치되어 있지 않습니다."

        # 비디오를 25fps로 변환
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_video_25fps = os.path.join(os.path.dirname(video_path), f"{base_name}_25fps.mp4")

        if progress_callback:
            progress_callback(0.2, "비디오를 25fps로 변환 중...")

        # FFmpeg로 25fps 변환
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-r", "25", temp_video_25fps
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return False, f"비디오 변환 실패: {result.stderr}"

        if progress_callback:
            progress_callback(0.3, "LatentSync 전처리 및 실행 중... (시간이 오래 걸릴 수 있습니다)")

        # 환경변수 설정
        env = os.environ.copy()
        env['PYTHONPATH'] = latentsync_dir

        # 전처리용 임시 디렉토리 생성
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        preprocess_dir = os.path.join(latentsync_dir, "temp", "preprocess", current_time)
        os.makedirs(preprocess_dir, exist_ok=True)

        # 1. 비디오를 전처리 디렉토리로 복사
        input_video_dir = os.path.join(preprocess_dir, "input")
        os.makedirs(input_video_dir, exist_ok=True)

        # 비디오 파일을 특정 이름으로 복사
        video_filename = f"{base_name}.mp4"
        preprocessed_input = os.path.join(input_video_dir, video_filename)
        shutil.copy2(temp_video_25fps, preprocessed_input)

        # 2. Affine Transform 실행
        affine_output_dir = os.path.join(preprocess_dir, "affine_transformed")
        os.makedirs(affine_output_dir, exist_ok=True)

        try:
            affine_cmd = [
                "python", "-m", "preprocess.affine_transform",
                "--input_dir", input_video_dir,
                "--output_dir", affine_output_dir,
                "--temp_dir", os.path.join(preprocess_dir, "temp"),
                "--resolution", "256",
                "--num_workers", "1"
            ]

            affine_result = subprocess.run(
                affine_cmd,
                cwd=latentsync_dir,
                capture_output=True,
                text=True,
                env=env,
                timeout=300
            )

            if affine_result.returncode == 0:
                # 전처리된 비디오 찾기
                processed_videos = []
                for root, dirs, files in os.walk(affine_output_dir):
                    for file in files:
                        if file.endswith('.mp4'):
                            processed_videos.append(os.path.join(root, file))

                if processed_videos:
                    abs_video_path = os.path.abspath(processed_videos[0])
                else:
                    abs_video_path = os.path.abspath(temp_video_25fps)
            else:
                abs_video_path = os.path.abspath(temp_video_25fps)

        except Exception:
            abs_video_path = os.path.abspath(temp_video_25fps)

        # 절대 경로로 변환
        abs_audio_path = os.path.abspath(audio_path)

        # 출력 파일 경로 생성 (임시 출력 경로)
        temp_output_path = os.path.join(latentsync_dir, "temp", f"output_{current_time}.mp4")

        # temp 디렉토리 생성
        os.makedirs(os.path.join(latentsync_dir, "temp"), exist_ok=True)

        # 체크포인트 파일 확인 (v1.6 우선 확인 - gradio_app.py와 동일한 로직)
        checkpoint_path_v16 = os.path.join(latentsync_dir, "checkpoints_v1.6", "latentsync_unet.pt")
        checkpoint_path_default = os.path.join(latentsync_dir, "checkpoints", "latentsync_unet.pt")

        if os.path.exists(checkpoint_path_v16):
            checkpoint_path = checkpoint_path_v16
        elif os.path.exists(checkpoint_path_default):
            checkpoint_path = checkpoint_path_default
        else:
            return False, f"LatentSync 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path_v16} 또는 {checkpoint_path_default}"

        # Whisper 모델 확인
        whisper_path = os.path.join(latentsync_dir, "checkpoints", "whisper", "tiny.pt")
        if not os.path.exists(whisper_path):
            return False, f"Whisper 모델을 찾을 수 없습니다: {whisper_path}"

        # 체크포인트 파일 크기로 버전 추정 (추론 최적화)
        checkpoint_size_gb = os.path.getsize(checkpoint_path) / (1024 ** 3)

        # 추론용 설정 파일 선택
        if checkpoint_size_gb > 3.0:  # 큰 모델 (1.6 버전 추정)
            config_path = os.path.join(latentsync_dir, "configs", "unet", "stage2_512.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(latentsync_dir, "configs", "unet", "stage2.yaml")
            expected_vram = "18GB"
        else:  # 작은 모델 (1.5 버전 추정)
            config_path = os.path.join(latentsync_dir, "configs", "unet", "stage2_efficient.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(latentsync_dir, "configs", "unet", "stage2.yaml")
            expected_vram = "8GB"

        if not os.path.exists(config_path):
            return False, "LatentSync 설정 파일을 찾을 수 없습니다"

        # 파라미터 최적화 (추론 시 성능 우선)
        inference_steps = "20"  # 기본값, 품질과 속도 균형
        guidance_scale = "1.5"  # 기본값, 안정성 우선

        # LatentSync 실행
        latentsync_cmd = [
            "python", "-m", "scripts.inference",
            "--unet_config_path", config_path,
            "--inference_ckpt_path", checkpoint_path,
            "--video_path", abs_video_path,
            "--audio_path", abs_audio_path,
            "--video_out_path", temp_output_path,
            "--inference_steps", inference_steps,
            "--guidance_scale", guidance_scale,
            "--seed", "1247",
            "--temp_dir", "temp",
            "--enable_deepcache"  # 추론 속도 향상
        ]

        # LatentSync 디렉토리에서 실행
        result = subprocess.run(
            latentsync_cmd,
            cwd=latentsync_dir,
            capture_output=True,
            text=True,
            env=env
        )

        if progress_callback:
            progress_callback(0.8, "립싱크 결과 처리 중...")

        if result.returncode != 0:
            # 얼굴 감지 실패 관련 에러 확인
            error_output = result.stderr.lower()
            if "face not detected" in error_output or "runtime error" in error_output:
                # 원본 비디오로 대체 출력 생성 (립싱크 없이)
                try:
                    # 원본 비디오와 새 오디오 합성
                    fallback_cmd = [
                        "ffmpeg", "-y", "-i", abs_video_path, "-i", abs_audio_path,
                        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                        "-shortest", output_path
                    ]

                    fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True)
                    if fallback_result.returncode == 0:
                        # 임시 파일 정리
                        if os.path.exists(temp_video_25fps):
                            os.remove(temp_video_25fps)

                        # 전처리 임시 디렉토리 정리
                        try:
                            if os.path.exists(preprocess_dir):
                                shutil.rmtree(preprocess_dir)
                        except:
                            pass  # 정리 실패해도 무시

                        return True, "얼굴 감지 실패로 립싱크 없이 오디오와 비디오가 합성되었습니다."
                    else:
                        return False, f"LatentSync 및 대체 처리 모두 실패: {result.stderr}"
                except Exception as e:
                    return False, f"대체 처리 중 오류: {e}"
            else:
                return False, f"LatentSync 실행 실패: {result.stderr}"

        # 결과 파일 확인
        if not os.path.exists(temp_output_path):
            return False, f"LatentSync 결과 파일이 생성되지 않았습니다: {temp_output_path}"

        # 결과 파일을 출력 경로로 복사
        shutil.copy2(temp_output_path, output_path)

        # 임시 파일 정리
        try:
            os.remove(temp_output_path)
        except:
            pass  # 정리 실패해도 무시

        if os.path.exists(temp_video_25fps):
            os.remove(temp_video_25fps)

        # 전처리 임시 디렉토리 정리
        try:
            if os.path.exists(preprocess_dir):
                shutil.rmtree(preprocess_dir)
        except:
            pass  # 정리 실패해도 무시

        if progress_callback:
            progress_callback(1.0, "립싱크 완료!")

        return True, f"립싱크가 적용된 비디오가 생성되었습니다: {output_path}"

    except Exception as e:
        return False, f"립싱크 처리 중 오류 발생: {str(e)}"


def update_vad_config(threshold, min_speech_duration_ms, max_speech_duration_s, min_silence_duration_ms, speech_pad_ms):
    """VAD 설정 업데이트 함수"""
    try:
        vad_config = {
            "threshold": float(threshold),
            "min_speech_duration_ms": int(min_speech_duration_ms),
            "max_speech_duration_s": float(max_speech_duration_s),
            "min_silence_duration_ms": int(min_silence_duration_ms),
            "speech_pad_ms": int(speech_pad_ms)
        }
        save_vad_config(vad_config)
        return f"✅ VAD 설정이 업데이트되었습니다!\n{json.dumps(vad_config, indent=2, ensure_ascii=False)}"
    except Exception as e:
        return f"❌ VAD 설정 업데이트 중 오류 발생: {str(e)}"


def load_current_vad_config():
    """현재 VAD 설정 로드"""
    try:
        config = load_vad_config()
        return (
            config.get("threshold", 0.5),
            config.get("min_speech_duration_ms", 3900),
            config.get("max_speech_duration_s", 20.0),
            config.get("min_silence_duration_ms", 250),
            config.get("speech_pad_ms", 100)
        )
    except:
        return 0.5, 3900, 20.0, 250, 100


def process_audio_video(
        input_file,
        enable_instruct,
        manual_command,
        command_mode,
        length_handling,
        overlap_handling,
        max_extension,
        enable_english,
        enable_chinese,
        enable_japanese,
        translation_length,
        translation_quality,
        enable_smart_compression,
        enable_speaker_diarization,
        speaker_mode,
        num_speakers,
        enable_speaker_splitting,
        enable_3sec_extension,
        vad_threshold,
        vad_min_speech_duration_ms,
        vad_max_speech_duration_s,
        vad_min_silence_duration_ms,
        vad_speech_pad_ms,
        enable_lip_sync,
        progress=gr.Progress()
):
    """메인 처리 함수"""
    if not input_file:
        return "❌ 파일을 선택해주세요."

    try:
        # VAD 설정 업데이트
        vad_config = {
            "threshold": float(vad_threshold),
            "min_speech_duration_ms": int(vad_min_speech_duration_ms),
            "max_speech_duration_s": float(vad_max_speech_duration_s),
            "min_silence_duration_ms": int(vad_min_silence_duration_ms),
            "speech_pad_ms": int(vad_speech_pad_ms)
        }
        save_vad_config(vad_config)

        # 설정 수집
        settings = {
            'enable_instruct': enable_instruct,
            'manual_command': manual_command if command_mode == 'manual' else None,
            'length_handling': length_handling,
            'overlap_handling': overlap_handling,
            'max_extension': int(max_extension),
            'selected_languages': [
                lang for lang, enabled in [
                    ('english', enable_english),
                    ('chinese', enable_chinese),
                    ('japanese', enable_japanese)
                ] if enabled
            ],
            'translation_length': float(translation_length),
            'quality_mode': translation_quality,
            'enable_smart_compression': enable_smart_compression,
            'enable_speaker_diarization': enable_speaker_diarization,
            'num_speakers': int(num_speakers) if speaker_mode == 'fixed' else None,
            'enable_speaker_splitting': enable_speaker_splitting,
            'enable_3sec_extension': enable_3sec_extension,
            'vocals_volume': 1.0,
            'background_volume': 0.8,
            'enable_lip_sync': enable_lip_sync
        }

        progress(0.1, desc="처리 시작...")

        # 파일 경로 처리
        input_path = input_file.name if hasattr(input_file, 'name') else input_file

        progress(0.3, desc="오디오 처리 중...")

        # 메인 처리 실행
        start_processing_with_settings(input_path, settings)

        progress(1.0, desc="처리 완료!")

        # 결과 파일 경로 생성
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(os.getcwd(), 'split_audio', base_name)

        result_info = f"✅ 처리 완료!\n\n"
        result_info += f"📁 출력 폴더: {output_dir}\n"
        result_info += f"🎤 VAD 설정: threshold={vad_threshold}, min_speech={vad_min_speech_duration_ms}ms\n"

        # 립싱크 활성화 시 LatentSync 결과 확인
        if enable_lip_sync:
            latentsync_dir = os.path.join(os.path.dirname(__file__), 'LatentSync')
            if os.path.exists(latentsync_dir):
                latentsync_output_dir = os.path.join(latentsync_dir, 'temp')
                if os.path.exists(latentsync_output_dir):
                    try:
                        latentsync_files = [f for f in os.listdir(latentsync_output_dir) if f.endswith('.mp4')]
                        if latentsync_files:
                            result_info += f"🎥 립싱크 결과 파일: {latentsync_files[0]}\n"
                    except (OSError, PermissionError):
                        # 디렉토리 접근 오류 시 무시
                        pass

        # 생성된 파일들 확인
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            result_info += f"📄 생성된 파일 개수: {len(files)}개\n"

            # SRT 파일 확인
            srt_files = [f for f in files if f.endswith('.srt')]
            if srt_files:
                result_info += f"📝 SRT 파일: {srt_files[0]}\n"

            # 음성 합성 결과 확인
            cosy_dir = os.path.join(output_dir, 'cosy_output')
            if os.path.exists(cosy_dir):
                cosy_files = os.listdir(cosy_dir)
                result_info += f"🎵 합성 음성 파일: {len(cosy_files)}개\n"

        return result_info

    except Exception as e:
        return f"❌ 처리 중 오류 발생: {str(e)}"


def generate_srt_only_func(input_file, progress=gr.Progress()):
    """SRT만 생성하는 함수"""
    if not input_file:
        return "❌ 파일을 선택해주세요."

    try:
        progress(0.1, desc="SRT 생성 시작...")

        input_path = input_file.name if hasattr(input_file, 'name') else input_file

        progress(0.5, desc="Whisper 처리 중...")

        # SRT 생성 실행
        srt_worker = generate_srt_only(input_path)
        srt_worker()

        progress(1.0, desc="SRT 생성 완료!")

        # 결과 파일 경로
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(os.getcwd(), 'split_audio', base_name)
        srt_path = os.path.join(output_dir, f"{base_name}.srt")

        result_info = f"✅ SRT 생성 완료!\n\n"
        result_info += f"📁 출력 폴더: {output_dir}\n"
        result_info += f"📝 SRT 파일: {srt_path}\n"

        return result_info

    except Exception as e:
        return f"❌ SRT 생성 중 오류 발생: {str(e)}"


def speaker_diarization_func(input_file, enable_speaker_diarization, speaker_mode, num_speakers,
                             progress=gr.Progress()):
    """화자 분리 함수"""
    if not input_file:
        return "❌ 파일을 선택해주세요."

    try:
        progress(0.1, desc="화자 분리 시작...")

        input_path = input_file.name if hasattr(input_file, 'name') else input_file

        settings = {
            'enable_speaker_diarization': enable_speaker_diarization,
            'speaker_mode': speaker_mode,
            'num_speakers': int(num_speakers) if speaker_mode == 'fixed' else None,
            'enable_speaker_splitting': False,
            'enable_3sec_extension': True,
        }

        progress(0.5, desc="화자 분리 처리 중...")

        test_speaker_diarization(input_path, settings)

        progress(1.0, desc="화자 분리 완료!")

        return "✅ 화자 분리 완료!"

    except Exception as e:
        return f"❌ 화자 분리 중 오류 발생: {str(e)}"


def merge_segments_func(input_file, merge_indices, synthesis_type, progress=gr.Progress()):
    """세그먼트 병합 함수"""
    if not input_file:
        return "❌ 파일을 선택해주세요."

    if not merge_indices.strip():
        return "❌ 병합할 세그먼트 번호를 입력해주세요."

    try:
        progress(0.1, desc="병합 시작...")

        input_path = input_file.name if hasattr(input_file, 'name') else input_file
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(os.getcwd(), 'split_audio', base_name)

        # SRT 파일에서 세그먼트 로드
        input_ext = os.path.splitext(input_path)[1]
        srt_path = os.path.join(output_dir, f"{base_name}{input_ext}.srt")

        if not os.path.exists(srt_path):
            return f"❌ SRT 파일을 찾을 수 없습니다: {srt_path}"

        segments = parse_srt_segments(srt_path)

        # 원본 오디오 로드
        orig_audio = AudioSegment.from_file(input_path)
        orig_dur = len(orig_audio)

        # 사용자 입력 파싱
        try:
            if merge_indices.lower() == 'all':
                selected_segments = segments
                merged_filename = f"{base_name}_{synthesis_type.lower()}_all_merged.wav"
            else:
                indices = json.loads(merge_indices)
                selected_segments = [segments[i - 1] for i in indices]
                merged_filename = f"{base_name}_{synthesis_type.lower()}_custom.wav"
        except (json.JSONDecodeError, IndexError, ValueError):
            return "❌ 잘못된 세그먼트 번호 형식입니다. 예: [1,2,3] 또는 'all'"

        progress(0.5, desc="세그먼트 병합 중...")

        # 소스 디렉토리 결정
        if synthesis_type == "Instruct2":
            source_dir = os.path.join(output_dir, 'cosy_output', 'instruct')
        else:  # Zero-shot
            source_dir = os.path.join(output_dir, 'cosy_output')

        merged_path = os.path.join(output_dir, merged_filename)

        # 병합 실행
        merge_segments_preserve_timing(
            selected_segments, orig_dur, source_dir, merged_path,
            length_handling="preserve",
            overlap_handling="fade",
            max_extension=50,
            enable_smart_compression=True
        )

        progress(1.0, desc="병합 완료!")

        return f"✅ 병합 완료!\n📁 출력 파일: {merged_path}"

    except Exception as e:
        return f"❌ 병합 중 오류 발생: {str(e)}"


def create_interface():
    """Gradio 인터페이스 생성"""

    # 현재 VAD 설정 로드
    default_vad_config = load_current_vad_config()

    with gr.Blocks(title="STT Voice Splitter - 웹 인터페이스", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎵 STT Voice Splitter - 웹 인터페이스")
        gr.Markdown("음성/영상 파일을 처리하여 텍스트 추출, 음성 합성, 다국어 번역을 수행합니다.")

        with gr.Tabs():
            # 메인 처리 탭
            with gr.TabItem("🎬 메인 처리"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_file = gr.File(
                            label="📁 입력 파일",
                            file_types=[".wav", ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"]
                        )

                        # Instruct2 설정
                        with gr.Group():
                            gr.Markdown("### 🎯 Instruct2 설정")
                            enable_instruct = gr.Checkbox(label="Instruct2 합성 활성화", value=False)
                            command_mode = gr.Radio(
                                choices=["auto", "manual"],
                                value="auto",
                                label="명령어 설정"
                            )
                            manual_command = gr.Textbox(
                                label="수동 명령어",
                                value="자연스럽게 말해",
                                placeholder="예: 자연스럽게 말해"
                            )

                        # VAD 설정 추가
                        with gr.Group():
                            gr.Markdown("### 🎤 VAD (음성 활동 감지) 설정")
                            vad_threshold = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=default_vad_config[0],
                                step=0.05,
                                label="VAD 임계값 (낮을수록 민감)"
                            )
                            vad_min_speech_duration_ms = gr.Slider(
                                minimum=500,
                                maximum=10000,
                                value=default_vad_config[1],
                                step=100,
                                label="최소 음성 길이 (ms)"
                            )
                            vad_max_speech_duration_s = gr.Slider(
                                minimum=5.0,
                                maximum=60.0,
                                value=default_vad_config[2],
                                step=1.0,
                                label="최대 음성 길이 (초)"
                            )
                            vad_min_silence_duration_ms = gr.Slider(
                                minimum=50,
                                maximum=1000,
                                value=default_vad_config[3],
                                step=50,
                                label="최소 무음 길이 (ms)"
                            )
                            vad_speech_pad_ms = gr.Slider(
                                minimum=0,
                                maximum=500,
                                value=default_vad_config[4],
                                step=10,
                                label="음성 패딩 (ms)"
                            )

                            # VAD 설정 업데이트 버튼
                            update_vad_btn = gr.Button("💾 VAD 설정 저장", variant="secondary", size="sm")
                            vad_update_output = gr.Textbox(label="VAD 설정 상태", lines=2, visible=False)

                            update_vad_btn.click(
                                fn=update_vad_config,
                                inputs=[vad_threshold, vad_min_speech_duration_ms, vad_max_speech_duration_s,
                                        vad_min_silence_duration_ms, vad_speech_pad_ms],
                                outputs=vad_update_output
                            ).then(
                                lambda: gr.update(visible=True),
                                outputs=vad_update_output
                            )

                        # 타임라인 설정
                        with gr.Group():
                            gr.Markdown("### ⏰ 타임라인 & 길이 처리")
                            length_handling = gr.Radio(
                                choices=["preserve", "fit"],
                                value="preserve",
                                label="합성 음성 길이 처리"
                            )
                            overlap_handling = gr.Radio(
                                choices=["fade", "cut"],
                                value="fade",
                                label="세그먼트 겹침 처리"
                            )
                            max_extension = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=10,
                                label="최대 확장율 (%)"
                            )
                            enable_lip_sync_checkbox = gr.Checkbox(label="🎥 립싱크 활성화", value=True)

                    with gr.Column(scale=1):
                        # 번역 설정
                        with gr.Group():
                            gr.Markdown("### 🌍 다국어 번역 설정")
                            enable_english = gr.Checkbox(label="🇺🇸 영어", value=True)
                            enable_chinese = gr.Checkbox(label="🇨🇳 중국어", value=True)
                            enable_japanese = gr.Checkbox(label="🇯🇵 일본어", value=True)
                            translation_length = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="번역 길이 비율"
                            )
                            translation_quality = gr.Radio(
                                choices=["concise", "balanced", "accurate"],
                                value="accurate",
                                label="번역 품질 우선순위"
                            )

                        # 기타 설정
                        with gr.Group():
                            gr.Markdown("### ⚙️ 기타 설정")
                            enable_smart_compression = gr.Checkbox(label="스마트 압축 활성화", value=True)
                            enable_speaker_diarization = gr.Checkbox(label="화자 분리 활성화", value=False)
                            speaker_mode = gr.Radio(
                                choices=["auto", "fixed"],
                                value="auto",
                                label="화자 수 설정"
                            )
                            num_speakers = gr.Slider(
                                minimum=2,
                                maximum=10,
                                value=2,
                                step=1,
                                label="고정 화자 수"
                            )
                            enable_speaker_splitting = gr.Checkbox(label="화자 기반 분할 활성화", value=False)
                            enable_3sec_extension = gr.Checkbox(label="3초 확장 활성화", value=True)

                process_btn = gr.Button("🚀 처리 시작", variant="primary", size="lg")
                main_output = gr.Textbox(label="📊 처리 결과", lines=10)

                process_btn.click(
                    fn=process_audio_video,
                    inputs=[
                        input_file, enable_instruct, manual_command, command_mode,
                        length_handling, overlap_handling, max_extension,
                        enable_english, enable_chinese, enable_japanese,
                        translation_length, translation_quality, enable_smart_compression,
                        enable_speaker_diarization, speaker_mode, num_speakers,
                        enable_speaker_splitting, enable_3sec_extension,
                        vad_threshold, vad_min_speech_duration_ms, vad_max_speech_duration_s,
                        vad_min_silence_duration_ms, vad_speech_pad_ms,
                        enable_lip_sync_checkbox
                    ],
                    outputs=main_output,
                    show_progress=True
                )

            # SRT 전용 생성 탭
            with gr.TabItem("📝 SRT 전용 생성"):
                srt_file = gr.File(
                    label="📁 오디오 파일",
                    file_types=[".wav", ".mp3"]
                )
                srt_btn = gr.Button("📝 SRT 생성", variant="primary")
                srt_output = gr.Textbox(label="📊 SRT 생성 결과", lines=5)

                srt_btn.click(
                    fn=generate_srt_only_func,
                    inputs=srt_file,
                    outputs=srt_output,
                    show_progress=True
                )

            # 화자 분리 탭
            with gr.TabItem("🗣️ 화자 분리"):
                speaker_file = gr.File(
                    label="📁 오디오 파일",
                    file_types=[".wav", ".mp3"]
                )
                with gr.Row():
                    with gr.Column():
                        speaker_diarization_enable = gr.Checkbox(label="화자 분리 활성화", value=True)
                        speaker_mode_dia = gr.Radio(
                            choices=["auto", "fixed"],
                            value="auto",
                            label="화자 수 설정"
                        )
                        num_speakers_dia = gr.Slider(
                            minimum=2,
                            maximum=10,
                            value=2,
                            step=1,
                            label="고정 화자 수"
                        )

                speaker_btn = gr.Button("🗣️ 화자 분리 실행", variant="primary")
                speaker_output = gr.Textbox(label="📊 화자 분리 결과", lines=5)

                speaker_btn.click(
                    fn=speaker_diarization_func,
                    inputs=[speaker_file, speaker_diarization_enable, speaker_mode_dia, num_speakers_dia],
                    outputs=speaker_output,
                    show_progress=True
                )

            # 세그먼트 병합 탭
            with gr.TabItem("🔗 세그먼트 병합"):
                merge_file = gr.File(
                    label="📁 처리된 파일 (원본 파일)",
                    file_types=[".wav", ".mp3", ".mp4"]
                )
                with gr.Row():
                    with gr.Column():
                        merge_indices = gr.Textbox(
                            label="병합할 세그먼트 번호",
                            placeholder="예: [1,2,3] 또는 'all'",
                            info="JSON 배열 형식으로 입력하거나 'all'로 전체 병합"
                        )
                        synthesis_type = gr.Radio(
                            choices=["Zero-shot", "Instruct2"],
                            value="Zero-shot",
                            label="합성 타입"
                        )

                merge_btn = gr.Button("🔗 병합 실행", variant="primary")
                merge_output = gr.Textbox(label="📊 병합 결과", lines=5)

                merge_btn.click(
                    fn=merge_segments_func,
                    inputs=[merge_file, merge_indices, synthesis_type],
                    outputs=merge_output,
                    show_progress=True
                )

            # VAD 설정 전용 탭 추가
            with gr.TabItem("🎤 VAD 설정"):
                gr.Markdown("## 🎤 VAD (Voice Activity Detection) 설정")
                gr.Markdown("음성 활동 감지 매개변수를 조정하여 음성 분할 품질을 개선할 수 있습니다.")

                with gr.Row():
                    with gr.Column():
                        vad_threshold_tab = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=default_vad_config[0],
                            step=0.05,
                            label="VAD 임계값",
                            info="낮을수록 더 많은 음성을 감지 (0.3-0.7 권장)"
                        )
                        vad_min_speech_duration_ms_tab = gr.Slider(
                            minimum=500,
                            maximum=10000,
                            value=default_vad_config[1],
                            step=100,
                            label="최소 음성 길이 (ms)",
                            info="이보다 짧은 음성은 무시"
                        )
                        vad_max_speech_duration_s_tab = gr.Slider(
                            minimum=5.0,
                            maximum=60.0,
                            value=default_vad_config[2],
                            step=1.0,
                            label="최대 음성 길이 (초)",
                            info="이보다 긴 음성은 분할"
                        )

                    with gr.Column():
                        vad_min_silence_duration_ms_tab = gr.Slider(
                            minimum=50,
                            maximum=1000,
                            value=default_vad_config[3],
                            step=50,
                            label="최소 무음 길이 (ms)",
                            info="음성 분할을 위한 최소 무음 구간"
                        )
                        vad_speech_pad_ms_tab = gr.Slider(
                            minimum=0,
                            maximum=500,
                            value=default_vad_config[4],
                            step=10,
                            label="음성 패딩 (ms)",
                            info="음성 앞뒤에 추가할 여백"
                        )

                        # 프리셋 버튼들
                        with gr.Row():
                            preset_sensitive_btn = gr.Button("🔍 민감 모드", variant="secondary")
                            preset_balanced_btn = gr.Button("⚖️ 균형 모드", variant="secondary")
                            preset_conservative_btn = gr.Button("🛡️ 보수적 모드", variant="secondary")

                save_vad_btn = gr.Button("💾 VAD 설정 저장", variant="primary", size="lg")
                vad_output_tab = gr.Textbox(label="📊 VAD 설정 결과", lines=5)

                # 프리셋 함수들
                def set_sensitive_preset():
                    return 0.3, 1000, 15.0, 100, 50

                def set_balanced_preset():
                    return 0.5, 3000, 20.0, 250, 100

                def set_conservative_preset():
                    return 0.7, 5000, 30.0, 500, 200

                preset_sensitive_btn.click(
                    fn=set_sensitive_preset,
                    outputs=[vad_threshold_tab, vad_min_speech_duration_ms_tab, vad_max_speech_duration_s_tab,
                             vad_min_silence_duration_ms_tab, vad_speech_pad_ms_tab]
                )

                preset_balanced_btn.click(
                    fn=set_balanced_preset,
                    outputs=[vad_threshold_tab, vad_min_speech_duration_ms_tab, vad_max_speech_duration_s_tab,
                             vad_min_silence_duration_ms_tab, vad_speech_pad_ms_tab]
                )

                preset_conservative_btn.click(
                    fn=set_conservative_preset,
                    outputs=[vad_threshold_tab, vad_min_speech_duration_ms_tab, vad_max_speech_duration_s_tab,
                             vad_min_silence_duration_ms_tab, vad_speech_pad_ms_tab]
                )

                save_vad_btn.click(
                    fn=update_vad_config,
                    inputs=[vad_threshold_tab, vad_min_speech_duration_ms_tab, vad_max_speech_duration_s_tab,
                            vad_min_silence_duration_ms_tab, vad_speech_pad_ms_tab],
                    outputs=vad_output_tab
                )

        # 사용법 안내
        with gr.Accordion("📖 사용법 안내", open=False):
            gr.Markdown("""
            ## 🎯 주요 기능
            
            1. **메인 처리**: 음성/영상 파일을 처리하여 텍스트 추출, 음성 합성, 다국어 번역 수행
            2. **SRT 전용 생성**: 음성 파일에서 SRT 자막 파일만 생성
            3. **화자 분리**: 여러 화자가 있는 음성을 분리하여 분석
            4. **세그먼트 병합**: 처리된 음성 세그먼트들을 선택적으로 병합
            5. **VAD 설정**: 음성 활동 감지 매개변수 조정으로 분할 품질 개선
            
            ## 📝 설정 가이드
            
            - **Instruct2**: 더 자연스러운 음성 합성을 위한 고급 모드
            - **길이 처리**: 합성된 음성의 길이를 원본에 맞춤 또는 보존
            - **번역 설정**: 원하는 언어로 번역 및 음성 합성
            - **화자 분리**: 여러 화자가 있는 음성을 개별적으로 처리
            - **VAD 설정**: 음성 감지 임계값 및 길이 제한 조정
            - **립싱크**: LatentSync을 사용하여 비디오에 립싱크 적용
            
            ## 🎤 VAD 설정 팁
            
            - **민감 모드**: 작은 소리도 감지, 짧은 음성 구간도 보존
            - **균형 모드**: 일반적인 용도에 적합한 기본 설정
            - **보수적 모드**: 명확한 음성만 감지, 노이즈 제거에 효과적
            
            ## 🔧 병합 방법
            
            - 특정 세그먼트 병합: `[1,2,3,5]` (세그먼트 1,2,3,5번 병합)
            - 전체 병합: `all` (모든 세그먼트 병합)
            """)

    return demo


if __name__ == "__main__":
    # Gradio 인터페이스 생성 및 실행
    demo = create_interface()

    # 서버 실행 (외부 접근 허용)
    demo.launch(
        server_name="0.0.0.0",  # 모든 IP에서 접근 허용
        server_port=7860,  # 포트 번호
        share=False,  # 공개 링크 생성 여부
        debug=True,  # 디버그 모드
        show_error=True  # 오류 표시
    )
