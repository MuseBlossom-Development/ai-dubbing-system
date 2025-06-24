import os
import threading
import time
from pydub import AudioSegment
from utils import log_message, is_video_file, is_audio_file
from video_processor import process_video_file, combine_processed_audio_with_background, combine_audio_with_video
from whisper_processor import run_full_whisper_processing, run_whisper_directory
from audio_processor import parse_srt_segments, merge_segments_preserve_timing, apply_speaker_based_splitting, \
    split_audio_by_srt, extend_short_segments_for_zeroshot, create_extended_segments_mapping
from batch_cosy import main as cosy_batch
from config import load_vad_config
from batch_translate import SUPPORTED_LANGUAGES


def apply_lip_sync_to_video(video_path, audio_path, output_path, frame_folder=None):
    """
    MuseTalk을 사용하여 립싱크 적용 (main_processor용)
    """
    try:
        import subprocess
        import shutil

        # MuseTalk 디렉토리 경로
        musetalk_dir = os.path.join(os.path.dirname(__file__), 'MuseTalk')

        if not os.path.exists(musetalk_dir):
            log_message("⚠️ MuseTalk이 설치되어 있지 않습니다. 립싱크를 건너뜁니다.")
            return False

        log_message("🎥 립싱크 처리 시작...")
        log_message(f"   📹 입력 비디오: {os.path.basename(video_path)}")
        log_message(f"   🎵 입력 오디오: {os.path.basename(audio_path)}")
        log_message(f"   💾 최종 출력 경로: {output_path}")
        log_message(f"   📂 MuseTalk 작업 디렉토리: {musetalk_dir}")
        log_message(f"   📁 MuseTalk 결과 생성 위치: {os.path.join(musetalk_dir, 'results', 'auto_lip_sync', 'v15')}")

        # 비디오를 25fps로 변환
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_video_25fps = os.path.join(os.path.dirname(video_path), f"{base_name}_temp_25fps.mp4")

        log_message("🔄 비디오를 25fps로 변환 중...")
        # FFmpeg로 25fps 변환
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-r", "25", temp_video_25fps
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            log_message(f"❌ 비디오 변환 실패: {result.stderr}")
            return False

        log_message("✅ 25fps 변환 완료")

        # MuseTalk 설정 파일 생성
        config_dir = os.path.join(musetalk_dir, 'configs', 'inference')
        os.makedirs(config_dir, exist_ok=True)

        config_path = os.path.join(config_dir, 'auto_lip_sync_config.yaml')

        # 상대 경로로 변환
        rel_video_path = os.path.relpath(temp_video_25fps, musetalk_dir)
        rel_audio_path = os.path.relpath(audio_path, musetalk_dir)

        # 개선된 설정 파일 (더 많은 파라미터 포함)
        config_content = f"""task_0:
  video_path: "{rel_video_path}"
  audio_path: "{rel_audio_path}"
  bbox_shift: 0
  extra_margin: 5
  parsing_mode: "default"
"""

        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)

        log_message("📝 MuseTalk 설정 파일 생성 완료")

        # 입력 파일 검증
        log_message("🔍 입력 파일 검증 중...")

        # 비디오 정보 확인
        try:
            import subprocess
            video_info_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams",
                              temp_video_25fps]
            video_info_result = subprocess.run(video_info_cmd, capture_output=True, text=True)
            if video_info_result.returncode == 0:
                import json
                video_info = json.loads(video_info_result.stdout)
                video_duration = float(video_info['format']['duration'])
                log_message(f"   📹 비디오 길이: {video_duration:.2f}초")
                for stream in video_info['streams']:
                    if stream['codec_type'] == 'video':
                        log_message(f"   📏 해상도: {stream['width']}x{stream['height']}")
                        log_message(f"   🎬 프레임레이트: {stream['r_frame_rate']}")
        except:
            log_message("   ⚠️ 비디오 정보 확인 실패")

        # 오디오 정보 확인
        try:
            audio_info_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", audio_path]
            audio_info_result = subprocess.run(audio_info_cmd, capture_output=True, text=True)
            if audio_info_result.returncode == 0:
                audio_info = json.loads(audio_info_result.stdout)
                audio_duration = float(audio_info['format']['duration'])
                log_message(f"   🎵 오디오 길이: {audio_duration:.2f}초")
                if abs(video_duration - audio_duration) > 1.0:
                    log_message(f"   ⚠️ 길이 차이: {abs(video_duration - audio_duration):.2f}초 (1초 이상 차이)")
        except:
            log_message("   ⚠️ 오디오 정보 확인 실패")

        log_message("🎬 MuseTalk 실행 중... (시간이 오래 걸릴 수 있습니다)")
        log_message("   - 얼굴 감지 및 랜드마크 추출")
        log_message("   - 오디오 특성 분석")
        log_message("   - 립싱크 프레임 생성")

        # MuseTalk 실행 (더 많은 파라미터 포함)
        musetalk_cmd = [
            "python", "-m", "scripts.inference",
            "--inference_config", "configs/inference/auto_lip_sync_config.yaml",
            "--result_dir", "results/auto_lip_sync",
            "--unet_model_path", "models/musetalkV15/unet.pth",
            "--unet_config", "models/musetalkV15/musetalk.json",
            "--version", "v15",
            "--bbox_shift", "0",
            "--extra_margin", "5"
        ]

        # MuseTalk 디렉토리에서 실행
        result = subprocess.run(
            musetalk_cmd,
            cwd=musetalk_dir,
            capture_output=True,
            text=True
        )

        # MuseTalk 실행 결과 상세 로그
        log_message(f"🔍 MuseTalk 실행 결과:")
        log_message(f"   반환 코드: {result.returncode}")

        if result.stdout:
            log_message(f"   표준 출력:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    log_message(f"     {line}")

        if result.stderr:
            log_message(f"   오류 출력:")
            for line in result.stderr.strip().split('\n'):
                if line.strip():
                    log_message(f"     {line}")

        if result.returncode != 0:
            log_message(f"❌ MuseTalk 실행 실패:")
            log_message(f"   반환 코드: {result.returncode}")
            if not result.stderr and not result.stdout:
                log_message("   출력 없음 - 환경 설정 문제일 수 있습니다")
            return False

        log_message("✅ MuseTalk 처리 완료")
        log_message("🔍 립싱크 결과 파일 검색 중...")

        # 결과 파일 찾기
        result_dir = os.path.join(musetalk_dir, 'results', 'auto_lip_sync', 'v15')

        log_message(f"   📁 결과 디렉토리: {result_dir}")

        if not os.path.exists(result_dir):
            log_message(f"❌ MuseTalk 결과 디렉토리를 찾을 수 없습니다:")
            log_message(f"   경로: {result_dir}")
            # 상위 디렉토리들 확인
            parent_dir = os.path.join(musetalk_dir, 'results')
            if os.path.exists(parent_dir):
                try:
                    subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
                    log_message(f"   results 내 서브디렉토리: {subdirs}")
                except:
                    log_message("   results 디렉토리 접근 불가")
            else:
                log_message("   results 디렉토리도 없음")
            return False

        # 생성된 파일 찾기
        try:
            all_files = os.listdir(result_dir)
            log_message(f"   📄 디렉토리 내 모든 파일: {all_files}")

            result_files = []
            for file in all_files:
                if file.endswith('.mp4'):
                    file_path = os.path.join(result_dir, file)
                    file_size = os.path.getsize(file_path)
                    result_files.append(file_path)
                    log_message(f"   🎥 MP4 파일 발견: {file} ({file_size} bytes)")
        except Exception as e:
            log_message(f"❌ 디렉토리 읽기 오류: {str(e)}")
            return False

        if not result_files:
            log_message("❌ 립싱크 결과 파일이 생성되지 않았습니다.")
            log_message("   가능한 원인:")
            log_message("   1. MuseTalk FFmpeg 합성 단계 실패")
            log_message("   2. 얼굴 감지 실패")
            log_message("   3. 오디오 길이와 비디오 길이 불일치")
            log_message("   4. 메모리 부족")
            log_message("   5. CUDA/GPU 문제")

            # 설정 파일 내용 확인
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_content_check = f.read()
                log_message("   📝 사용된 설정:")
                for line in config_content_check.strip().split('\n'):
                    log_message(f"     {line}")
            except:
                log_message("   설정 파일 읽기 실패")

            return False

        log_message(f"✅ 립싱크 결과 파일 발견: {os.path.basename(result_files[0])}")

        # 첫 번째 결과 파일을 출력 경로로 복사
        log_message("📁 최종 출력 파일로 복사 중...")
        log_message(f"   🔄 복사: {result_files[0]}")
        log_message(f"   📍 목적지: {output_path}")
        shutil.copy2(result_files[0], output_path)

        # MuseTalk 결과 파일 정리 (선택사항)
        try:
            os.remove(result_files[0])
            log_message("🗑️ MuseTalk 임시 결과 파일 정리 완료")
        except:
            log_message("⚠️ MuseTalk 임시 파일 정리 실패 (권한 문제일 수 있음)")

        # 임시 파일 정리
        if os.path.exists(temp_video_25fps):
            os.remove(temp_video_25fps)
            log_message("🗑️ 25fps 임시 파일 정리 완료")

        log_message(f"✅ 립싱크 완료!")
        log_message(f"   📁 최종 위치: {output_path}")
        log_message(f"   📊 파일 크기: {os.path.getsize(output_path) / (1024 * 1024):.1f}MB")
        return True

    except Exception as e:
        log_message(f"❌ 립싱크 처리 중 오류 발생: {str(e)}")
        import traceback
        log_message(f"   상세 오류: {traceback.format_exc()}")
        return False


def process_complete_pipeline(input_file, settings):
    """
    완전한 영상 처리 파이프라인

    플로우:
    1. 영상에서 음성 추출
    2. 음성에서 보컬/배경음 분리
    3. 보컬만 STT → 번역 → TTS → 병합
    4. 처리된 보컬과 배경음 재합성
    5. 최종 영상과 음성 합성
    """
    try:
        # 디버그: 설정 확인
        log_message(f"🔧 처리 설정 확인:")
        log_message(f"  - 립싱크 활성화: {settings.get('enable_lip_sync', False)}")
        log_message(f"  - 선택된 언어: {settings.get('selected_languages', [])}")

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_base_dir = os.path.join(os.getcwd(), 'video_output', base_name)
        os.makedirs(output_base_dir, exist_ok=True)

        log_message(f"🎬 영상 처리 파이프라인 시작: {input_file}")

        # Step 1: 영상 처리 (음성 추출 + 보컬/배경음 분리)
        log_message("📹 Step 1: 영상에서 음성 추출 및 보컬/배경음 분리")
        extracted_audio, vocals_path, background_path, original_video = process_video_file(
            input_file, output_base_dir
        )

        if not vocals_path or not background_path:
            log_message("❌ 보컬/배경음 분리 실패, 파이프라인 중단")
            return

        # Step 2: 보컬 파일로 STT 처리
        log_message("🎤 Step 2: 보컬 음성으로 STT 처리")
        vad_config = load_vad_config()
        output_dir, segments, orig_duration = run_full_whisper_processing(vocals_path, vad_config)

        if not output_dir or not segments:
            log_message("❌ STT 처리 실패, 파이프라인 중단")
            return

        # 화자 기반 분할 적용 (활성화된 경우)
        if settings.get('enable_speaker_splitting', False):
            log_message("🗣️ 화자 기반 분할 적용 중...")
            base_name = os.path.splitext(os.path.basename(vocals_path))[0]
            srt_path = os.path.join(output_dir, f"{base_name}{os.path.splitext(vocals_path)[1]}.srt")
            segments, orig_duration = apply_speaker_based_splitting(
                vocals_path,
                srt_path,
                output_dir,
                True
            )

        if not output_dir or not segments:
            log_message("❌ 화자 분할 처리 실패")
            return

        # Step 3: 텍스트 번역 및 음성 합성
        log_message("🔄 Step 3: 텍스트 번역 및 음성 합성")

        # 번역 설정 구성
        translation_settings = {
            'translation_length': settings.get('translation_length', 0.8),
            'quality_mode': settings.get('quality_mode', 'balanced'),
            'selected_languages': settings.get('selected_languages', ['english'])
        }

        # Whisper 디렉토리 처리 (번역 포함)
        selected_languages = run_whisper_directory(output_dir, translation_settings)

        if not selected_languages:
            log_message("❌ 번역 처리 실패, 파이프라인 중단")
            return

        # 각 언어별로 음성 합성
        processed_vocals = {}

        for lang in selected_languages:
            lang_name = SUPPORTED_LANGUAGES[lang]['name'].lower()
            trans_type = "free"
            text_dir = os.path.join(output_dir, 'txt', lang_name, trans_type)

            if not os.path.exists(text_dir):
                log_message(f"⏭️ {SUPPORTED_LANGUAGES[lang]['name']} {trans_type} 텍스트 없음, 건너뛰기")
                continue

            log_message(f"🔊 {SUPPORTED_LANGUAGES[lang]['name']} ({trans_type}) 음성 합성 시작...")

            # CosyVoice2 합성 호출
            cosy_out = os.path.join(output_dir, 'cosy_output', lang_name, trans_type)
            os.makedirs(cosy_out, exist_ok=True)

            # 3초 미만 세그먼트 확장 (제로샷 개선용)
            if settings.get('enable_3sec_extension', True):
                log_message("🔄 제로샷 합성을 위한 3초 미만 세그먼트 확장 중...")
                from audio_processor import extend_short_segments_for_zeroshot, create_extended_segments_mapping
                extended_wav_dir = extend_short_segments_for_zeroshot(output_dir, min_duration_ms=3000)

                if extended_wav_dir:
                    # 확장 매핑 정보 생성
                    mapping_info = create_extended_segments_mapping(output_dir, extended_wav_dir)
                    log_message(f"📊 세그먼트 확장 완료: {len(mapping_info.get('segments_info', []))}개 파일 처리")

                    # 확장된 세그먼트를 사용하여 합성
                    synthesis_audio_dir = extended_wav_dir
                else:
                    # 확장 실패 시 원본 사용
                    log_message("⚠️ 세그먼트 확장 실패, 원본 세그먼트 사용")
                    synthesis_audio_dir = os.path.join(output_dir, 'wav')
            else:
                # 설정 비활성화 시 원본 사용
                synthesis_audio_dir = os.path.join(output_dir, 'wav')

            # 메모리 정리 (합성 전)
            import gc
            gc.collect()

            try:
                # UI 설정값 적용
                enable_instruct = settings.get('enable_instruct', False)
                manual_command = settings.get('manual_command', None)

                log_message(f"  언어: {SUPPORTED_LANGUAGES[lang]['name']}")
                log_message(f"  번역 유형: {trans_type}")
                log_message(f"  Instruct2 활성화: {enable_instruct}")
                if manual_command:
                    log_message(f"  수동 명령어: {manual_command}")

                # CosyVoice2 배치 합성 (언어 정보 포함)
                cosy_batch(
                    audio_dir=synthesis_audio_dir,
                    prompt_text_dir=os.path.join(output_dir, 'txt', 'ko'),
                    text_dir=text_dir,
                    out_dir=cosy_out,
                    enable_instruct=enable_instruct,
                    manual_command=manual_command,
                    target_language=lang
                )

                log_message(f"✅ {SUPPORTED_LANGUAGES[lang]['name']} ({trans_type}) 합성 완료")

                # 실제 합성 파일들은 zero_shot 서브디렉토리에 저장됨
                actual_synthesis_dir = os.path.join(cosy_out, 'zero_shot')

                # 병합
                merged_path = os.path.join(output_dir, f"{base_name}_{lang_name}_merged.wav")
                merge_segments_preserve_timing(
                    segments,
                    orig_duration,  # 이미 밀리초 단위이므로 * 1000 제거
                    actual_synthesis_dir,  # zero_shot 서브디렉토리 참조
                    merged_path,
                    length_handling=settings.get('length_handling', 'preserve'),
                    overlap_handling=settings.get('overlap_handling', 'fade'),
                    max_extension=settings.get('max_extension', 50),
                    enable_smart_compression=settings.get('enable_smart_compression', True)
                )

                processed_vocals[lang] = merged_path
                log_message(f"✅ {lang_name} 보컬 병합 완료: {merged_path}")

            except Exception as e:
                log_message(f"❌ {SUPPORTED_LANGUAGES[lang]['name']} 처리 오류: {e}")
                continue

            # 합성 완료 후 메모리 정리
            gc.collect()

        # Step 4: 각 언어별로 보컬+배경음 합성 및 최종 영상 생성
        log_message("🎵 Step 4: 보컬과 배경음 합성 및 최종 영상 생성")

        final_videos = []

        for lang_code, processed_vocal_path in processed_vocals.items():
            lang_name = SUPPORTED_LANGUAGES[lang_code]['name'].lower()
            lang_display_name = SUPPORTED_LANGUAGES[lang_code]['name']

            # 보컬 + 배경음 합성
            combined_audio_path = os.path.join(output_base_dir, f"{base_name}_{lang_name}_combined.wav")

            success = combine_processed_audio_with_background(
                processed_vocal_path,
                background_path,
                combined_audio_path,
                vocals_volume=settings.get('vocals_volume', 1.0),
                background_volume=settings.get('background_volume', 0.8)
            )

            if success:
                # 최종 영상 생성
                final_video_path = os.path.join(output_base_dir, f"{base_name}_{lang_name}_final.mp4")

                video_success = combine_audio_with_video(
                    original_video,
                    combined_audio_path,
                    final_video_path
                )

                if video_success:
                    final_videos.append((lang_name, final_video_path))
                    log_message(f"✅ {lang_name} 최종 영상 완료: {final_video_path}")
                else:
                    log_message(f"❌ {lang_name} 영상 합성 실패")
            else:
                log_message(f"❌ {lang_name} 음성 합성 실패")

        # Step 5: 립싱크 처리 (모든 언어에 대해)
        if settings.get('enable_lip_sync', False):
            log_message("🗣️ 립싱크 처리 활성화됨. 모든 언어 버전에 립싱크 적용 중...")
            log_message(f"   🎯 처리 대상 언어: {len(processed_vocals)}개")
            lip_sync_videos = []

            for idx, (lang_code, processed_vocal_path) in enumerate(processed_vocals.items(), 1):
                lang_name = SUPPORTED_LANGUAGES[lang_code]['name'].lower()
                lang_display_name = SUPPORTED_LANGUAGES[lang_code]['name']

                # 해당 언어의 기본 최종 영상 경로
                regular_video_path = os.path.join(output_base_dir, f"{base_name}_{lang_name}_final.mp4")

                # 립싱크용 출력 경로 생성
                lip_sync_output_path = os.path.join(output_base_dir, f"{base_name}_{lang_name}_final_lipsynced.mp4")

                log_message(f"🎬 [{idx}/{len(processed_vocals)}] {lang_display_name} 립싱크 처리 시작...")
                log_message(f"   📹 기본 영상: {os.path.basename(regular_video_path)}")
                log_message(f"   🎵 음성 파일: {os.path.basename(processed_vocal_path)}")

                # 기본 영상이 존재하는지 확인
                if not os.path.exists(regular_video_path):
                    log_message(f"❌ {lang_display_name} 기본 영상 파일을 찾을 수 없습니다:")
                    log_message(f"   경로: {regular_video_path}")
                    continue

                # 립싱크 처리 (기본 영상 + 해당 언어 음성)
                log_message(f"⚙️ {lang_display_name} MuseTalk 처리 중...")
                start_time = time.time()

                if apply_lip_sync_to_video(regular_video_path, processed_vocal_path, lip_sync_output_path):
                    end_time = time.time()
                    processing_time = int(end_time - start_time)

                    lip_sync_videos.append((lang_name, lip_sync_output_path))
                    log_message(f"✅ {lang_display_name} 립싱크 완료! (소요시간: {processing_time}초)")
                    log_message(f"   💾 출력: {os.path.basename(lip_sync_output_path)}")
                else:
                    end_time = time.time()
                    processing_time = int(end_time - start_time)
                    log_message(f"❌ {lang_display_name} 립싱크 처리 실패 (소요시간: {processing_time}초)")

                # 진행률 표시
                if len(processed_vocals) > 1:
                    progress_percent = (idx / len(processed_vocals)) * 100
                    log_message(f"📊 립싱크 진행률: {progress_percent:.1f}% ({idx}/{len(processed_vocals)})")

            # 립싱크 결과 요약
            if lip_sync_videos:
                log_message("🎉 모든 언어 립싱크 영상 생성 완료:")
                log_message(f"   ✅ 성공: {len(lip_sync_videos)}개")
                log_message(f"   ❌ 실패: {len(processed_vocals) - len(lip_sync_videos)}개")
                for lang_name, video_path in lip_sync_videos:
                    log_message(f"   🗣️ {lang_name} 립싱크: {os.path.basename(video_path)}")
            else:
                log_message("❌ 모든 언어 립싱크 영상 생성 실패")
                log_message("   원인: 기본 영상 또는 음성 파일 누락, MuseTalk 처리 오류 등")

        # 완료 메시지
        if final_videos:
            log_message("🎉 영상 처리 파이프라인 완료!")
            log_message("📁 생성된 최종 영상들:")
            for lang_name, video_path in final_videos:
                log_message(f"   🎬 {lang_name} (기본): {video_path}")

            # 립싱크 영상이 있다면 함께 표시
            if settings.get('enable_lip_sync', False) and 'lip_sync_videos' in locals():
                log_message("📁 생성된 립싱크 영상들:")
                for lang_name, video_path in lip_sync_videos:
                    log_message(f"   🗣️ {lang_name} (립싱크): {os.path.basename(video_path)}")
        else:
            log_message("❌ 최종 영상 생성 실패")

    except Exception as e:
        log_message(f"영상 처리 파이프라인 오류: {e}")
        import traceback
        log_message(f"상세 오류: {traceback.format_exc()}")


def process_audio_only_pipeline(input_file, settings):
    """기존 음성 파일 처리 파이프라인 (원본 기능 유지)"""
    try:
        log_message("🎵 음성 파일 처리 파이프라인 시작")

        vad_config = load_vad_config()
        output_dir, segments, orig_duration = run_full_whisper_processing(input_file, vad_config)

        if not output_dir or not segments:
            log_message("❌ STT 처리 실패")
            return

        # 화자 기반 분할 적용 (활성화된 경우)
        if settings.get('enable_speaker_splitting', False):
            log_message("🗣️ 화자 기반 분할 적용 중...")
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            srt_path = os.path.join(output_dir, f"{base_name}{os.path.splitext(input_file)[1]}.srt")
            segments, total_duration = apply_speaker_based_splitting(
                input_file,
                srt_path,
                output_dir,
                True
            )
        else:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            srt_path = os.path.join(output_dir, f"{base_name}{os.path.splitext(input_file)[1]}.srt")
            segments = parse_srt_segments(srt_path)
            orig_audio = AudioSegment.from_file(input_file)
            original_duration_ms = len(orig_audio)

        if not output_dir or not segments:
            log_message("❌ 화자 분할 처리 실패")
            return

        # 번역 설정 구성
        translation_settings = {
            'translation_length': settings.get('translation_length', 0.8),
            'quality_mode': settings.get('quality_mode', 'balanced'),
            'selected_languages': settings.get('selected_languages', ['english'])
        }

        # Whisper 디렉토리 처리 (번역 포함)
        selected_languages = run_whisper_directory(output_dir, translation_settings)

        if not selected_languages:
            log_message("❌ 번역 처리 실패")
            return

        # 각 언어별로 음성 합성 및 병합
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        input_ext = os.path.splitext(input_file)[1]
        srt_path = os.path.join(output_dir, f"{base_name}{input_ext}.srt")
        segments = parse_srt_segments(srt_path)
        orig_audio = AudioSegment.from_file(input_file)
        original_duration_ms = len(orig_audio)

        for lang in selected_languages:
            lang_name = SUPPORTED_LANGUAGES[lang]['name'].lower()
            trans_type = "free"
            text_dir = os.path.join(output_dir, 'txt', lang_name, trans_type)

            if not os.path.exists(text_dir):
                continue

            cosy_out = os.path.join(output_dir, 'cosy_output', lang_name, trans_type)
            os.makedirs(cosy_out, exist_ok=True)

            # 3초 미만 세그먼트 확장 (제로샷 개선용)
            if settings.get('enable_3sec_extension', True):
                log_message("🔄 제로샷 합성을 위한 3초 미만 세그먼트 확장 중...")
                from audio_processor import extend_short_segments_for_zeroshot, create_extended_segments_mapping
                extended_wav_dir = extend_short_segments_for_zeroshot(output_dir, min_duration_ms=3000)

                if extended_wav_dir:
                    # 확장 매핑 정보 생성
                    mapping_info = create_extended_segments_mapping(output_dir, extended_wav_dir)
                    log_message(f"📊 세그먼트 확장 완료: {len(mapping_info.get('segments_info', []))}개 파일 처리")

                    # 확장된 세그먼트를 사용하여 합성
                    synthesis_audio_dir = extended_wav_dir
                else:
                    # 확장 실패 시 원본 사용
                    log_message("⚠️ 세그먼트 확장 실패, 원본 세그먼트 사용")
                    synthesis_audio_dir = os.path.join(output_dir, 'wav')
            else:
                # 설정 비활성화 시 원본 사용
                synthesis_audio_dir = os.path.join(output_dir, 'wav')

            # CosyVoice2 합성
            try:
                cosy_batch(
                    audio_dir=synthesis_audio_dir,
                    prompt_text_dir=os.path.join(output_dir, 'txt', 'ko'),
                    text_dir=text_dir,
                    out_dir=cosy_out,
                    enable_instruct=settings.get('enable_instruct', False),
                    manual_command=settings.get('manual_command', None),
                    target_language=lang
                )

                log_message(f"✅ {lang_name} ({trans_type}) 합성 완료")

                # 실제 합성 파일들은 zero_shot 서브디렉토리에 저장됨
                actual_synthesis_dir = os.path.join(cosy_out, 'zero_shot')

                # 병합
                merged_path = os.path.join(output_dir, f"{base_name}_{lang_name}_merged.wav")
                merge_segments_preserve_timing(
                    segments,
                    original_duration_ms,
                    actual_synthesis_dir,  # zero_shot 서브디렉토리 참조
                    merged_path,
                    length_handling=settings.get('length_handling', 'preserve'),
                    overlap_handling=settings.get('overlap_handling', 'fade'),
                    max_extension=settings.get('max_extension', 50),
                    enable_smart_compression=settings.get('enable_smart_compression', True)
                )

                log_message(f"✅ {lang_name} 처리 완료: {merged_path}")

            except Exception as e:
                log_message(f"❌ {lang_name} 처리 오류: {e}")

        log_message("🎵 음성 파일 처리 완료")

    except Exception as e:
        log_message(f"음성 파일 처리 오류: {e}")


def start_processing_with_settings(input_file, settings):
    """파일 타입에 따른 처리 분기"""

    def worker():
        try:
            if is_video_file(input_file):
                log_message("🎬 영상 파일 감지 - 영상 처리 파이프라인 시작")
                process_complete_pipeline(input_file, settings)
            elif is_audio_file(input_file):
                log_message("🎵 음성 파일 감지 - 음성 처리 파이프라인 시작")
                process_audio_only_pipeline(input_file, settings)
            else:
                log_message("❌ 지원하지 않는 파일 형식입니다")
        except Exception as e:
            log_message(f"처리 중 오류 발생: {e}")

    threading.Thread(target=worker, daemon=True).start()
