import os
import shutil
import gc  # 메모리 정리를 위한 가비지 컬렉션
from config import get_whisper_cli_path, get_model_path, resource_path, load_vad_config, IS_MACOS
from utils import log_message, run_command_with_logging
from audio_processor import split_audio_by_srt, parse_srt_segments
from batch_translate import batch_translate, SUPPORTED_LANGUAGES


def cleanup_whisper_memory():
    """Whisper 처리 후 메모리 정리"""
    log_message("🧹 Whisper 메모리 정리 중...")

    # 가비지 컬렉션 실행
    for _ in range(3):
        collected = gc.collect()
        log_message(f"   가비지 컬렉션: {collected}개 객체 해제")

    # CUDA 캐시 비우기 (PyTorch가 있는 경우)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log_message("   CUDA 캐시 정리 완료")
    except ImportError:
        pass  # torch가 없으면 무시

    log_message("✅ Whisper 메모리 정리 완료")


def generate_srt_only(input_file):
    """SRT 파일만 생성하는 함수"""
    base = os.path.splitext(os.path.basename(input_file))[0]
    out = os.path.join(os.getcwd(), 'split_audio', base)
    os.makedirs(out, exist_ok=True)

    vad_config = load_vad_config()

    def srt_worker():
        try:
            log_message('== SRT 전용 생성 시작 ==')
            model_path, is_coreml = get_model_path()
            log_message(f'사용 모델: {model_path} (CoreML: {is_coreml})')

            whisper_cli = get_whisper_cli_path()
            whisper_cmd = [
                whisper_cli,
                '--vad',
                '--vad-model', resource_path('whisper.cpp/models/ggml-silero-v5.1.2.bin'),
                '--vad-threshold', str(vad_config['threshold']),
                '--vad-min-speech-duration-ms', str(vad_config['min_speech_duration_ms']),
                '--vad-min-silence-duration-ms', str(vad_config['min_silence_duration_ms']),
                '--vad-max-speech-duration-s', str(vad_config['max_speech_duration_s']),
                '--vad-speech-pad-ms', str(vad_config['speech_pad_ms']),
                '-f', input_file,
                '-m', model_path,
                '--output-srt',
                '--language', 'ko',
            ]

            run_command_with_logging(whisper_cmd, cwd=os.path.dirname(input_file),
                                     description="SRT 전용 생성")

            # SRT 파일을 출력 디렉토리로 이동
            input_dir = os.path.dirname(input_file)
            moved = False
            for f in os.listdir(input_dir):
                if f.startswith(base) and f.lower().endswith('.srt'):
                    src_path = os.path.join(input_dir, f)
                    dst_path = os.path.join(out, f)
                    shutil.move(src_path, dst_path)
                    log_message(f'✅ SRT 파일 생성 완료: {dst_path}')
                    moved = True
                    break

            if not moved:
                log_message('❌ SRT 파일 생성 실패: 파일을 찾을 수 없습니다.')
            else:
                log_message('== SRT 전용 생성 완료 ==')
                # SRT 생성 후 메모리 정리
                cleanup_whisper_memory()

        except Exception as e:
            log_message(f'SRT 생성 오류: {e}')

    return srt_worker


def run_whisper_directory(output_dir: str, translation_settings=None):
    """개별 세그먼트 텍스트 처리 (이미 전체 처리에서 생성됨 - 건너뛰기)"""
    log_message("🚀 개별 세그먼트 텍스트는 이미 생성됨 - 번역 단계로 진행")

    base = os.path.basename(output_dir)
    txt_root = os.path.join(output_dir, 'txt')
    ko_folder = os.path.join(txt_root, 'ko')
    os.makedirs(ko_folder, exist_ok=True)

    # 기존에 생성된 TXT 파일을 ko 폴더로 정리
    txt_file = None
    for f in os.listdir(output_dir):
        if f.lower().endswith('.txt'):
            txt_file = os.path.join(output_dir, f)
            break

    if txt_file and os.path.exists(txt_file):
        # 전체 텍스트 파일을 읽어서 세그먼트별로 분할
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # 세그먼트별 텍스트 생성 (간단한 분할)
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        wav_folder = os.path.join(output_dir, 'wav')
        wav_files = sorted([f for f in os.listdir(wav_folder)
                            if f.startswith(f"{base}_") and f.endswith('.wav')])

        # 각 세그먼트에 대응하는 텍스트 생성
        for i, wav_file in enumerate(wav_files):
            name = os.path.splitext(wav_file)[0]
            ko_file = os.path.join(ko_folder, f"{name}.ko.txt")

            # 대응하는 텍스트 라인이 있으면 사용, 없으면 빈 문자열
            text_content = lines[i] if i < len(lines) else ""

            with open(ko_file, 'w', encoding='utf-8') as f:
                f.write(text_content)

            if text_content:
                log_message(f"한국어 텍스트 저장: {os.path.basename(ko_file)}")
    else:
        log_message("⚠️ 전체 텍스트 파일을 찾을 수 없음 - 빈 텍스트로 진행")

    # 번역 설정 처리
    if translation_settings is None:
        translation_settings = {
            'translation_length': 0.7,
            'quality_mode': 'balanced',
            'selected_languages': ['english']
        }

    translation_length = translation_settings.get('translation_length', 0.8)
    quality_mode = translation_settings.get('quality_mode', 'balanced')
    selected_languages = translation_settings.get('selected_languages', ['english'])

    log_message(f"번역 대상 언어: {', '.join(selected_languages)}")
    log_message(f"번역 설정 - 길이 비율: {translation_length}, 품질 모드: {quality_mode}")

    # 다국어 번역 실행
    try:
        batch_translate(
            input_dir=ko_folder,
            output_dir=txt_root,
            length_ratio=translation_length,
            target_languages=selected_languages
        )
        log_message("✅ 다국어 번역 완료")
        log_message("🧹 Gemma3 모델 메모리 해제 완료 - CosyVoice 합성 준비")
    except Exception as e:
        log_message(f"❌ 번역 오류: {e}")
        return

    return selected_languages


def run_full_whisper_processing(input_file, vad_config=None):
    """전체 Whisper 처리 파이프라인 - SRT와 텍스트를 한 번에 생성"""
    base = os.path.splitext(os.path.basename(input_file))[0]
    out = os.path.join(os.getcwd(), 'split_audio', base)
    os.makedirs(out, exist_ok=True)

    if vad_config is None:
        vad_config = load_vad_config()

    log_message('== whisper.cpp 실행 (VAD+SRT+TXT) ==')
    log_message(f'🔧 VAD 설정 확인:')
    log_message(f'   threshold: {vad_config["threshold"]}')
    log_message(f'   min_speech_duration_ms: {vad_config["min_speech_duration_ms"]}')
    log_message(f'   max_speech_duration_s: {vad_config["max_speech_duration_s"]}')
    log_message(f'   min_silence_duration_ms: {vad_config["min_silence_duration_ms"]}')
    log_message(f'   speech_pad_ms: {vad_config["speech_pad_ms"]}')

    model_path, is_coreml = get_model_path()
    log_message(f'사용 모델: {model_path} (CoreML: {is_coreml})')

    whisper_cli = get_whisper_cli_path()
    whisper_cmd = [
        whisper_cli,
        '--vad',
        '--vad-model', resource_path('whisper.cpp/models/ggml-silero-v5.1.2.bin'),
        '--vad-threshold', str(vad_config['threshold']),
        '--vad-min-speech-duration-ms', str(vad_config['min_speech_duration_ms']),
        '--vad-min-silence-duration-ms', str(vad_config['min_silence_duration_ms']),
        '--vad-max-speech-duration-s', str(vad_config['max_speech_duration_s']),
        '--vad-speech-pad-ms', str(vad_config['speech_pad_ms']),
        '-f', input_file,
        '-m', model_path,
        '--output-srt',
        '--output-txt',  # 텍스트도 함께 생성
        '--language', 'ko',
    ]

    log_message(f'🔧 Whisper 명령어: {" ".join(whisper_cmd)}')

    # CoreML은 자동으로 사용됨
    if is_coreml and IS_MACOS:
        log_message('CoreML 모델 사용 중 (자동 가속)')

    run_command_with_logging(whisper_cmd, cwd=os.path.dirname(input_file),
                             description="whisper.cpp VAD+SRT+TXT 처리")
    log_message('== SRT+TXT 생성 완료 ==')

    # 메모리 정리
    cleanup_whisper_memory()

    # 입력 디렉토리에서 생성된 파일들 확인
    input_dir = os.path.dirname(input_file)
    log_message(f'🔍 입력 디렉토리({input_dir})에서 생성된 파일들:')

    generated_files = []
    for f in os.listdir(input_dir):
        if f.startswith(base) and (f.lower().endswith('.srt') or f.lower().endswith('.txt')):
            file_path = os.path.join(input_dir, f)
            file_size = os.path.getsize(file_path)
            log_message(f'   📄 {f} ({file_size} bytes)')
            generated_files.append(f)

    # SRT 파일을 출력 디렉토리로 이동
    moved_srt = None
    for f in generated_files:
        src_path = os.path.join(input_dir, f)
        dst_path = os.path.join(out, f)

        # 기존 파일이 있으면 삭제 후 이동
        if os.path.exists(dst_path):
            os.remove(dst_path)
            log_message(f'🗑️ 기존 파일 삭제: {f}')

        shutil.move(src_path, dst_path)
        log_message(f'📁 파일 이동: {f} → {dst_path}')

        if f.lower().endswith('.srt'):
            moved_srt = f

    log_message(f"📂 출력 디렉토리 내용: {os.listdir(out)}")

    srt_files = [f for f in os.listdir(out) if f.lower().endswith('.srt')]
    if not srt_files:
        log_message('❌ 에러: SRT 파일을 찾을 수 없습니다.')
        return None, None, None

    srt_path = os.path.join(out, srt_files[0])
    log_message(f'🎯 WAV 분할에 사용할 SRT 파일: {srt_files[0]}')
    log_message(f'   경로: {srt_path}')

    # SRT 파일 내용 미리보기
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_preview = f.read()[:1000]  # 처음 1000자만
            log_message(f'📝 SRT 파일 내용 미리보기:\n{srt_preview}...')
    except Exception as e:
        log_message(f'⚠️ SRT 파일 읽기 오류: {e}')

    segments, orig_dur = split_audio_by_srt(input_file, srt_path, out)
    log_message(f'== {len(segments)}개 세그먼트 분할 완료 ==')

    return out, segments, orig_dur
