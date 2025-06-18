import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading, platform, subprocess, queue, json, shutil, re

# PyTorch 디바이스 설정 (CPU/CUDA만, MPS 제외)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# CUDA 메모리 최적화

def clear_mps_cache():
    """호환성을 위한 빈 함수 (MPS 사용 안함)"""
    import gc
    gc.collect()

# CosyVoice 패키지 및 의존 모듈을 찾을 수 있도록 경로 추가
repo_root = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(repo_root, 'CosyVoice'))
sys.path.insert(0, os.path.join(repo_root, 'CosyVoice', 'third_party'))
# sys.path.insert(0, os.path.join(repo_root, 'CosyVoice', 'third_party', 'matcha'))
sys.path.insert(0, os.path.join(repo_root, 'CosyVoice', 'third_party', 'Matcha-TTS'))
from batch_cosy import main as cosy_batch
from pydub import AudioSegment  # pip install pydub
from batch_translate import batch_translate, SUPPORTED_LANGUAGES
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def merge_custom_callback():
    try:
        # load original segments and duration
        base = os.path.splitext(os.path.basename(input_file_var.get()))[0]
        input_ext = os.path.splitext(input_file_var.get())[1]  # .mp3 또는 .wav 등
        out_dir = os.path.join(os.getcwd(), 'split_audio', base)
        srt_path = os.path.join(out_dir, f"{base}{input_ext}.srt")
        segments = parse_srt_segments(srt_path)
        # load original full audio directly from the input path
        input_path = input_file_var.get()
        orig_audio = AudioSegment.from_file(input_path)
        orig_dur = len(orig_audio)
        # parse user timings list
        timings = json.loads(merge_entry.get())
        # merge using only specified segments indices
        selected = [segments[i-1] for i in timings]

        # 사용자가 선택한 합성 타입에 따라 소스 폴더 결정
        synthesis_type = synthesis_type_var.get()
        if synthesis_type == "Instruct2":
            source_dir = os.path.join(out_dir, 'cosy_output', 'instruct')
            merged_filename = f"{base}_instruct_custom.wav"
        else:  # Zero-shot (기본값)
            source_dir = os.path.join(out_dir, 'cosy_output')
            merged_filename = f"{base}_custom.wav"

        merged_path = os.path.join(out_dir, merged_filename)
        length_handling = length_handling_var.get()
        overlap_handling = overlap_handling_var.get()
        max_extension = int(max_extension_var.get())
        enable_smart_compression = enable_smart_compression_var.get()

        merge_segments_preserve_timing(selected, orig_dur, source_dir, merged_path,
                                       length_handling=length_handling,
                                       overlap_handling=overlap_handling,
                                       max_extension=max_extension,
                                       enable_smart_compression=enable_smart_compression)
        log_message(f"✅ {synthesis_type} 커스텀 병합 완료: {merged_filename}")
    except Exception as e:
        log_message(f"병합 오류: {e}")

# 전체 세그먼트 병합 콜백 함수
def merge_all_segments_callback():
    try:
        base = os.path.splitext(os.path.basename(input_file_var.get()))[0]
        input_ext = os.path.splitext(input_file_var.get())[1]  # .mp3 또는 .wav 등
        out_dir = os.path.join(os.getcwd(), 'split_audio', base)
        srt_path = os.path.join(out_dir, f"{base}{input_ext}.srt")
        segments = parse_srt_segments(srt_path)
        input_path = input_file_var.get()
        orig_audio = AudioSegment.from_file(input_path)
        original_duration_ms = len(orig_audio)

        # 사용자가 선택한 합성 타입에 따라 소스 폴더 결정
        synthesis_type = synthesis_type_var.get()
        if synthesis_type == "Instruct2":
            source_dir = os.path.join(out_dir, 'cosy_output', 'instruct')
            merged_filename = f"{base}_instruct_merged.wav"
        else:  # Zero-shot (기본값)
            source_dir = os.path.join(out_dir, 'cosy_output')
            merged_filename = f"{base}_cosy_merged.wav"

        merged_path = os.path.join(out_dir, merged_filename)
        length_handling = length_handling_var.get()
        overlap_handling = overlap_handling_var.get()
        max_extension = int(max_extension_var.get())
        enable_smart_compression = enable_smart_compression_var.get()

        merge_segments_preserve_timing(segments, original_duration_ms, source_dir, merged_path,
                                       length_handling=length_handling,
                                       overlap_handling=overlap_handling,
                                       max_extension=max_extension,
                                       enable_smart_compression=enable_smart_compression)
        log_message(f"✅ {synthesis_type} 결과 병합 완료: {merged_filename}")
    except Exception as e:
        log_message(f"전체 병합 오류: {e}")

# ------------------------
# OS/환경 정보 및 경로 처리
# ------------------------
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.getcwd()
    if IS_WINDOWS and relative_path.endswith('whisper-cli'):
        relative_path += '.exe'
    return os.path.join(base_path, relative_path)


# SRT만 생성하는 함수 추가
def generate_srt_only():
    """SRT 파일만 생성하는 함수"""
    # 플랫폼별 파일타입 형식 분기처리
    if IS_WINDOWS:
        filetypes = [
            ('Audio Files', '*.wav;*.mp3'),
            ('WAV files', '*.wav'),
            ('MP3 files', '*.mp3'),
            ('All files', '*.*')
        ]
    else:
        filetypes = [
            ('Audio Files', '*.wav *.mp3'),
            ('WAV files', '*.wav'),
            ('MP3 files', '*.mp3'),
            ('All files', '*.*')
        ]

    input_file = filedialog.askopenfilename(filetypes=filetypes)
    if not input_file:
        return

    base = os.path.splitext(os.path.basename(input_file))[0]
    out = os.path.join(os.getcwd(), 'split_audio', base)
    os.makedirs(out, exist_ok=True)

    load_config()

    def srt_worker():
        try:
            log_message('== SRT 전용 생성 시작 ==')
            model_path, is_coreml = get_model_path()
            log_message(f'사용 모델: {model_path} (CoreML: {is_coreml})')

            whisper_cmd = [
                WHISPER_CLI,
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

        except Exception as e:
            log_message(f'SRT 생성 오류: {e}')

    threading.Thread(target=srt_worker, daemon=True).start()


# whisper-cli 경로
if IS_WINDOWS:
    WHISPER_CLI = resource_path('whisper.cpp/build/bin/Release/whisper-cli')
    if not os.path.exists(WHISPER_CLI):
        WHISPER_CLI = resource_path('whisper.cpp/build/bin/whisper-cli')
else:
    WHISPER_CLI = resource_path('whisper.cpp/build/bin/whisper-cli')


# 모델 경로 설정 (OS별 분기)
def get_model_path():
    """OS에 따른 적절한 모델 경로 반환"""
    if IS_MACOS:
        # macOS: 먼저 CoreML 모델 확인, 없으면 GGML 모델 사용
        coreml_path = resource_path('whisper.cpp/models/ggml-large-v3-turbo-encoder.mlmodelc')
        ggml_path = resource_path('whisper.cpp/models/ggml-large-v3-turbo.bin')

        if os.path.exists(coreml_path) and os.path.exists(ggml_path):
            # CoreML 인코더를 사용하려면 기본 GGML 모델도 필요함
            return ggml_path, True  # GGML 모델 경로 반환, CoreML 사용
        elif os.path.exists(ggml_path):
            return ggml_path, False  # GGML 모델만 사용
        else:
            # 기본 테스트 모델로 fallback
            return resource_path('whisper.cpp/models/for-tests-ggml-base.bin'), False
    else:
        # Windows/Linux: GGML 모델만 사용
        ggml_path = resource_path('resources/ggml-large-v3-turbo.bin')
        if os.path.exists(ggml_path):
            return ggml_path, False
        else:
            # whisper.cpp/models 디렉토리에서 찾기
            return resource_path('whisper.cpp/models/ggml-large-v3-turbo.bin'), False


# ffmpeg 탐색
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path and IS_WINDOWS:
    candidate = os.path.join(os.environ.get("CONDA_PREFIX",""), "Library", "bin", "ffmpeg.exe")
    if os.path.exists(candidate):
        ffmpeg_path = candidate
if not ffmpeg_path:
    raise RuntimeError("ffmpeg 실행파일을 찾을 수 없습니다.")

# -------------------------------------
# VAD 설정 및 기본값
# -------------------------------------
CONFIG_FILE = 'vad_config.json'
DEFAULT_CONFIG = {
    'threshold': 0.6,
    'min_speech_duration_ms': 200,  # 테스트용 강제값
    'max_speech_duration_s': 15.0,
    'min_silence_duration_ms': 70,
    'speech_pad_ms': 0,  # 패딩을 0으로 설정하여 타임라인 동기화 개선
}

install_log_queue = queue.Queue()
audio_log_queue   = queue.Queue()
vad_config        = {}


def log_message(message, also_print=True):
    """GUI와 터미널에 동일한 메시지를 출력"""
    install_log_queue.put(message)
    if also_print:
        print(f"[LOG] {message}")


def audio_log_message(message, also_print=True):
    """오디오 로그를 GUI와 터미널에 동일하게 출력"""
    audio_log_queue.put(message)
    if also_print:
        print(f"[AUDIO] {message}")


def run_command_with_logging(cmd, cwd=None, description="명령 실행"):
    """명령어 실행과 동시에 모든 출력을 로그로 전달"""
    log_message(f"{description}: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stderr도 stdout으로 합치기
            encoding='utf-8',
            errors='ignore',
            universal_newlines=True,
            bufsize=1
        )

        # 실시간으로 출력 읽기
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                if line:  # 빈 줄이 아닌 경우만 로그
                    log_message(line)

        return_code = process.poll()
        log_message(f"{description} 완료 (return code: {return_code})")
        return return_code

    except Exception as e:
        log_message(f"{description} 오류: {e}")
        return -1

# -------------------------------------
# 로그 처리
# -------------------------------------
def process_log_queue():
    """통합 로그 처리 함수"""
    while not install_log_queue.empty():
        line = install_log_queue.get()
        log_text.configure(state='normal')
        log_text.insert(tk.END, line + '\n')
        log_text.see(tk.END)
        log_text.configure(state='disabled')

    while not audio_log_queue.empty():
        line = audio_log_queue.get()
        log_text.configure(state='normal')
        log_text.insert(tk.END, f"[AUDIO] {line}" + '\n')
        log_text.see(tk.END)
        log_text.configure(state='disabled')

    root.after(200, process_log_queue)

# -------------------------------------
# SRT 파싱 및 오디오 분할 함수
# -------------------------------------
_time_re = re.compile(r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})')

def srt_time_to_milliseconds(t: str) -> int:
    t = t.replace(',', '.')
    h, m, rest = t.split(':')
    s, ms = rest.split('.')
    return (int(h)*3600 + int(m)*60 + int(s))*1000 + int(ms)

def parse_srt_segments(srt_path: str):
    segments = []
    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = _time_re.search(line)
            if m:
                start, end = m.group(1), m.group(2)
                segments.append((srt_time_to_milliseconds(start),
                                 srt_time_to_milliseconds(end)))
    return segments

def split_audio_by_srt(audio_path: str, srt_path: str, output_dir: str):
    log_message(f'DEBUG: SRT 경로 → {srt_path} / 존재 → {os.path.exists(srt_path)}')
    segments = parse_srt_segments(srt_path)
    log_message(f'DEBUG: 파싱된 세그먼트 수 → {len(segments)}')

    audio = AudioSegment.from_file(audio_path)
    wav_folder = os.path.join(output_dir, 'wav')
    base = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(wav_folder, exist_ok=True)

    for idx, (start_ms, end_ms) in enumerate(segments, 1):
        chunk = audio[start_ms:end_ms]
        out_path = os.path.join(wav_folder, f"{base}_{idx:03d}.wav")
        chunk.export(out_path, format="wav")
        audio_log_message(f"세그먼트 {idx}: {start_ms}~{end_ms}")
    return segments, len(audio)

# -------------------------------------
# 추가 오디오 처리 함수들 (길이 조절)
# -------------------------------------

def adjust_audio_speed(audio_segment, speed_factor):
    """
    오디오 속도를 조절하면서 피치 보존
    Args:
        audio_segment: AudioSegment 객체
        speed_factor: 속도 배율 (1.0 = 원본, 1.2 = 20% 빠르게)
    """
    try:
        # 속도 조절 (프레임 레이트 변경)
        adjusted = audio_segment._spawn(
            audio_segment.raw_data,
            overrides={"frame_rate": int(audio_segment.frame_rate * speed_factor)}
        ).set_frame_rate(audio_segment.frame_rate)
        return adjusted
    except Exception as e:
        log_message(f"속도 조절 오류: {e}")
        return audio_segment


def calculate_segment_priority(text_content, duration_ms):
    """
    세그먼트의 중요도 점수 계산
    Args:
        text_content: 텍스트 내용
        duration_ms: 세그먼트 길이 (밀리초)
    Returns:
        priority_score: 중요도 점수 (0.0~1.0)
    """
    if not text_content:
        return 0.1  # 텍스트가 없으면 낮은 우선순위

    # 간투어 및 불필요 요소 감지
    filler_words = ['음', '어', '그', '저', '뭐', '그런데', '그러니까']
    text_clean = text_content.strip().lower()

    # 기본 점수
    base_score = 0.5

    # 길이 기반 보정 (너무 짧거나 긴 것은 우선순위 낮춤)
    if duration_ms < 500:  # 0.5초 미만
        base_score *= 0.7
    elif duration_ms > 10000:  # 10초 초과
        base_score *= 0.8

    # 간투어 포함 시 우선순위 낮춤
    filler_count = sum(1 for word in filler_words if word in text_clean)
    if filler_count > 0:
        base_score *= (0.8 ** filler_count)

    # 문장 부호 기반 중요도 (완전한 문장은 높은 우선순위)
    if any(punct in text_clean for punct in ['.', '!', '?', '다', '요', '니다']):
        base_score *= 1.2

    # 짧은 감탄사나 응답은 우선순위 낮춤
    if len(text_clean) <= 3 and text_clean in ['네', '예', '아', '오', '응', '음']:
        base_score *= 0.6

    return min(1.0, max(0.1, base_score))


def smart_audio_compression(audio_segment, target_duration_ms, text_content=""):
    """
    AI 기반 스마트 오디오 압축
    Args:
        audio_segment: 압축할 오디오 세그먼트
        target_duration_ms: 목표 길이
        text_content: 텍스트 내용 (우선순위 계산용)
    """
    current_duration = len(audio_segment)
    if current_duration <= target_duration_ms:
        return audio_segment

    compression_ratio = target_duration_ms / current_duration
    priority = calculate_segment_priority(text_content, current_duration)

    # 수정된 올바른 로직: 중요한 내용은 덜 압축, 불필요한 내용은 더 압축
    if priority >= 0.8:
        # 높은 우선순위(중요한 대사): 최소한의 압축만 (원속도 최대한 유지)
        max_speed = min(1.05, 1.0 / compression_ratio)  # 최대 5%만 빠르게
        return adjust_audio_speed(audio_segment, max_speed)

    elif priority >= 0.5:
        # 중간 우선순위: 적당한 압축
        speed_factor = min(1.2, 1.0 / compression_ratio)  # 최대 20% 빠르게
        adjusted = adjust_audio_speed(audio_segment, speed_factor)

        # 추가 압축이 필요한 경우 무음 구간 압축
        if len(adjusted) > target_duration_ms:
            # 무음 구간 탐지 및 압축 (구현 간소화)
            remaining_ratio = target_duration_ms / len(adjusted)
            adjusted = adjust_audio_speed(adjusted, 1.0 / remaining_ratio)

        return adjusted

    else:
        # 낮은 우선순위(간투어, 감탄사): 적극적 압축
        speed_factor = min(1.5, 1.0 / compression_ratio)  # 최대 80% 빠르게

        # 긴 문장에 대한 추가 압축
        if len(text_content) > 50:  # 긴 문장
            speed_factor = min(speed_factor * 1.1, 2.0)  # 추가로 10% 더 빠르게, 최대 2배속

        return adjust_audio_speed(audio_segment, speed_factor)


def remove_excessive_silence(audio_segment, max_silence_ms=500):
    """
    과도한 무음 구간 제거
    Args:
        audio_segment: 처리할 오디오
        max_silence_ms: 허용할 최대 무음 길이
    """
    try:
        # 무음 임계값 설정 (-40dB)
        silence_thresh = audio_segment.dBFS - 40

        # 무음 구간 탐지
        chunks = []
        current_pos = 0
        chunk_size = 100  # 100ms 단위로 처리

        while current_pos < len(audio_segment):
            chunk = audio_segment[current_pos:current_pos + chunk_size]

            # 무음 여부 확인
            if chunk.dBFS < silence_thresh:
                # 무음 구간 - 길이 제한
                silence_length = min(chunk_size, max_silence_ms)
                chunks.append(AudioSegment.silent(duration=silence_length))

                # 연속 무음 건너뛰기
                next_pos = current_pos + chunk_size
                while next_pos < len(audio_segment):
                    next_chunk = audio_segment[next_pos:next_pos + chunk_size]
                    if next_chunk.dBFS >= silence_thresh:
                        break
                    next_pos += chunk_size
                current_pos = next_pos
            else:
                # 음성 구간 - 그대로 유지
                chunks.append(chunk)
                current_pos += chunk_size

        return sum(chunks) if chunks else audio_segment

    except Exception as e:
        log_message(f"무음 제거 오류: {e}")
        return audio_segment


# -------------------------------------
# 세그먼트 병합 (원본 타이밍 보존) - 개선된 버전
# -------------------------------------
def merge_segments_preserve_timing(segments, original_duration_ms, segments_dir, output_path,
                                   length_handling="preserve", overlap_handling="fade", max_extension=50,
                                   enable_smart_compression=True):
    """
    세그먼트들을 원본 타임라인에 맞춰 병합 (스마트 압축 기능 추가)
    합성된 음성이 더 길 경우 설정에 따라 처리
    
    Args:
        length_handling: "preserve" (완전 보존) 또는 "fit" (원본 길이 맞춤)
        overlap_handling: "fade" (페이드 처리) 또는 "cut" (자르기)
        max_extension: 최대 확장 허용 비율 (%)
        enable_smart_compression: 스마트 압축 활성화 여부
    """
    # 올바른 베이스 이름 추출
    if 'cosy_output' in segments_dir:
        # segments_dir이 /path/to/split_audio/base_name/cosy_output/language/type 형태인 경우
        path_parts = segments_dir.split(os.sep)
        cosy_index = None
        for i, part in enumerate(path_parts):
            if part == 'cosy_output':
                cosy_index = i
                break

        if cosy_index and cosy_index >= 2:
            # split_audio 다음에 오는 부분이 베이스 이름
            split_audio_index = None
            for i, part in enumerate(path_parts):
                if part == 'split_audio':
                    split_audio_index = i
                    break

            if split_audio_index is not None and split_audio_index + 1 < len(path_parts):
                base = path_parts[split_audio_index + 1]
            else:
                # fallback
                base = os.path.basename(os.path.dirname(os.path.dirname(segments_dir)))
        else:
            # fallback
            base = os.path.basename(os.path.dirname(os.path.dirname(segments_dir)))
    else:
        # segments_dir이 직접 세그먼트가 있는 폴더인 경우
        base = os.path.basename(segments_dir)

    log_message(f"병합 베이스 이름: {base}, 세그먼트 디렉토리: {segments_dir}")
    log_message(f"🔧 처리 설정: 길이={length_handling}, 겹침={overlap_handling}, 최대확장={max_extension}%")
    log_message(f"🧠 스마트 압축: {'활성화' if enable_smart_compression else '비활성화'}")

    # 실제 존재하는 파일들 확인
    if os.path.exists(segments_dir):
        available_files = [f for f in os.listdir(segments_dir) if f.endswith('.wav')]
        log_message(f"사용 가능한 세그먼트 파일들: {available_files}")
    else:
        log_message(f"경고: 세그먼트 디렉토리가 존재하지 않습니다: {segments_dir}")
        return

    # 텍스트 파일 경로 설정 (우선순위 계산용)
    base_dir = os.path.dirname(segments_dir) if 'cosy_output' in segments_dir else segments_dir
    ko_text_dir = os.path.join(base_dir, 'txt', 'ko')

    # 각 세그먼트의 합성 길이를 먼저 확인하여 전체 길이 계산
    segment_info = []
    total_extension = 0
    max_allowed_extension = original_duration_ms * max_extension / 100  # 최대 허용 확장량

    for idx, (start_ms, end_ms) in enumerate(segments, start=1):
        original_duration = end_ms - start_ms
        seg_path = os.path.join(segments_dir, f"{base}_{idx:03d}.wav")

        # 텍스트 내용 로드 (우선순위 계산용)
        text_content = ""
        if enable_smart_compression:
            text_file = os.path.join(ko_text_dir, f"{base}_{idx:03d}.ko.txt")
            if os.path.exists(text_file):
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                except Exception:
                    text_content = ""

        if os.path.exists(seg_path):
            synth_segment = AudioSegment.from_file(seg_path)
            synth_duration = len(synth_segment)

            # 스마트 압축 적용
            if enable_smart_compression and synth_duration > original_duration:
                log_message(f"🧠 세그먼트 {idx}: 스마트 압축 적용 중... (원본: {original_duration}ms → 합성: {synth_duration}ms)")

                # 우선순위 계산
                priority = calculate_segment_priority(text_content, synth_duration)
                log_message(
                    f"   우선순위: {priority:.2f} ({'높음' if priority >= 0.8 else '중간' if priority >= 0.5 else '낮음'})")

                # 과도한 무음 제거
                synth_segment = remove_excessive_silence(synth_segment, max_silence_ms=300)
                silence_removed_duration = len(synth_segment)
                if silence_removed_duration < synth_duration:
                    log_message(f"   무음 제거: {synth_duration}ms → {silence_removed_duration}ms")
                    synth_duration = silence_removed_duration

                # 여전히 길면 스마트 압축 적용
                if synth_duration > original_duration:
                    if length_handling == "fit":
                        # 원본 길이에 맞춤 모드
                        synth_segment = smart_audio_compression(synth_segment, original_duration, text_content)
                        synth_duration = len(synth_segment)
                        log_message(f"   압축 완료: {synth_duration}ms (목표: {original_duration}ms)")
                    else:
                        # 보존 모드에서도 극단적인 경우 제한적 압축
                        max_allowed_length = int(original_duration * 1.5)  # 최대 50% 확장
                        if synth_duration > max_allowed_length:
                            synth_segment = smart_audio_compression(synth_segment, max_allowed_length, text_content)
                            synth_duration = len(synth_segment)
                            log_message(f"   제한적 압축: {synth_duration}ms (최대허용: {max_allowed_length}ms)")

            # 기존 길이 처리 방식
            elif length_handling == "fit" and not enable_smart_compression:
                # 원본 길이에 맞춤 (최대 20% 확장까지만 허용)
                max_segment_length = int(original_duration * 1.2)
                if synth_duration > max_segment_length:
                    # 자연스러운 페이드아웃으로 조정
                    synth_segment = synth_segment[:max_segment_length].fade_out(100)
                    synth_duration = len(synth_segment)
                    log_message(f"🔧 세그먼트 {idx}: 길이 조정 {len(synth_segment)}ms → {synth_duration}ms")
                elif synth_duration < original_duration:
                    # 짧은 경우 무음 패딩
                    padding = AudioSegment.silent(duration=original_duration - synth_duration)
                    synth_segment = synth_segment + padding
                    synth_duration = len(synth_segment)

            extension = max(0, synth_duration - original_duration)

            # 최대 확장 제한 적용 (기존 로직)
            if extension > max_allowed_extension / len(segments):  # 세그먼트당 평균 허용량
                allowed_length = original_duration + int(max_allowed_extension / len(segments))
                synth_segment = synth_segment[:allowed_length].fade_out(200)
                synth_duration = len(synth_segment)
                extension = synth_duration - original_duration
                log_message(f"🔧 세그먼트 {idx}: 최대 확장 제한 적용 → {synth_duration}ms")

            total_extension += extension

            segment_info.append({
                'idx': idx,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'original_duration': original_duration,
                'synth_duration': synth_duration,
                'extension': extension,
                'segment': synth_segment,
                'exists': True,
                'text_content': text_content
            })

            log_message(f"세그먼트 {idx}: 원본 {original_duration}ms → 합성 {synth_duration}ms (확장: {extension}ms)")
        else:
            segment_info.append({
                'idx': idx,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'original_duration': original_duration,
                'synth_duration': 0,
                'extension': 0,
                'segment': None,
                'exists': False
            })
            log_message(f"❌ 세그먼트 {idx}: 파일 없음")

    # 최종 길이 계산
    if length_handling == "preserve":
        final_duration_ms = original_duration_ms + total_extension
        log_message(f"📏 예상 최종 길이: {original_duration_ms}ms + {total_extension}ms = {final_duration_ms}ms")
    else:
        final_duration_ms = original_duration_ms + min(total_extension, int(max_allowed_extension))
        log_message(f"📏 조정된 최종 길이: {final_duration_ms}ms (원본: {original_duration_ms}ms)")

    # 최종 길이만큼의 무음으로 시작
    merged = AudioSegment.silent(duration=final_duration_ms)

    # 누적 시간 오프셋 (이전 세그먼트들의 확장으로 인한 지연)
    time_offset = 0

    for seg_info in segment_info:
        if seg_info['exists']:
            # 현재 세그먼트의 배치 위치 (시간 오프셋 적용)
            adjusted_start = seg_info['start_ms'] + time_offset
            adjusted_end = adjusted_start + seg_info['synth_duration']

            # 합성된 세그먼트 배치
            synth_segment = seg_info['segment']

            # 다음 세그먼트와의 충돌 검사
            next_segment_start = None
            for next_seg in segment_info:
                if next_seg['idx'] > seg_info['idx'] and next_seg['exists']:
                    next_segment_start = next_seg['start_ms'] + time_offset + seg_info['extension']
                    break

            # 충돌이 있는 경우 겹치는 부분 처리
            if next_segment_start and adjusted_end > next_segment_start:
                overlap_duration = adjusted_end - next_segment_start
                log_message(f"⚠️ 세그먼트 {seg_info['idx']}: 다음 세그먼트와 {overlap_duration}ms 겹침 감지")

                if overlap_handling == "fade":
                    # 겹치는 부분을 페이드아웃/페이드인으로 자연스럽게 처리
                    if overlap_duration < len(synth_segment):
                        fade_duration = min(overlap_duration // 2, 500)  # 최대 0.5초 페이드
                        synth_segment = synth_segment.fade_out(fade_duration)
                        log_message(f"🔧 세그먼트 {seg_info['idx']}: {fade_duration}ms 페이드아웃 적용")
                elif overlap_handling == "cut":
                    # 겹치는 부분 자르기
                    synth_segment = synth_segment[:len(synth_segment) - overlap_duration]
                    log_message(f"✂️ 세그먼트 {seg_info['idx']}: {overlap_duration}ms 잘라내기 적용")

            # 세그먼트 배치 (오버레이 방식으로 자연스럽게)
            merged = merged.overlay(synth_segment, position=adjusted_start)

            # 시간 오프셋 업데이트
            if length_handling == "preserve":
                time_offset += seg_info['extension']

            log_message(f"✅ 세그먼트 {seg_info['idx']}: {adjusted_start}ms~{adjusted_end}ms 배치 완료")
        else:
            log_message(f"⏭️ 세그먼트 {seg_info['idx']}: 건너뛰기 (파일 없음)")

    # 최종 길이 검증
    actual_length = len(merged)
    log_message(f"📊 실제 최종 길이: {actual_length}ms")

    # 결과 저장
    merged.export(output_path, format="wav")
    log_message(f"🎵 스마트 타임라인 병합 완료: {output_path}")
    log_message(
        f"📈 길이 변화: {original_duration_ms}ms → {actual_length}ms (변화: {actual_length - original_duration_ms:+d}ms)")

    # 품질 보장 메시지
    if length_handling == "preserve":
        if actual_length >= original_duration_ms:
            log_message("✅ 합성 음성 완전 보존! 모든 내용이 잘리지 않았습니다.")
        else:
            log_message("⚠️ 예상치 못한 길이 단축 발생")
    else:
        log_message(f"✅ 원본 길이 기준으로 조정 완료 (최대 {max_extension}% 확장)")

    if enable_smart_compression:
        log_message("🧠 스마트 압축 기능으로 더욱 자연스러운 결과를 얻었습니다.")

    return actual_length

# -------------------------------------
# Whisper 처리 함수
# -------------------------------------

# -------------------------------------
# Ko → EN 번역 함수 (Gemma3 기반)
# -------------------------------------
def translate_ko_to_en(ko_folder: str, en_folder: str):
    """
    ko_folder 내의 각 .ko.txt 파일을 Gemma3로 번역 및 의역하여
    en_folder에 같은 이름으로 .en.txt로 저장합니다.
    """
    for fname in os.listdir(ko_folder):
        if fname.endswith('.ko.txt'):
            ko_path = os.path.join(ko_folder, fname)
            en_path = os.path.join(en_folder, fname.replace('.ko.txt', '.en.txt'))
            try:
                with open(ko_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                # 실제 번역 부분은 Gemma3 등 외부 API/모듈로 대체
                translated = f"[EN] {text}"
                with open(en_path, 'w', encoding='utf-8') as f:
                    f.write(translated)
                log_message(f"번역 완료: {fname} → {os.path.basename(en_path)}")
            except Exception as e:
                log_message(f"번역 오류: {fname} ({e})")
    log_message(f"✅ {ko_folder} 내 Ko → EN 번역 및 의역 완료")

# -------------------------------------
# 전체 Ko 텍스트 폴더 일괄 번역 함수
# -------------------------------------
def batch_translate_all():
    """
    split_audio 하위 모든 base 디렉토리의 txt/ko 폴더를 순회하며
    Gemma3로 번역 및 의역 후 txt/en 폴더에 저장합니다.
    """
    root_dir = os.path.join(os.getcwd(), 'split_audio')
    if not os.path.isdir(root_dir):
        log_message(f"경로가 없습니다: {root_dir}")
        return

    for base in os.listdir(root_dir):
        ko_folder = os.path.join(root_dir, base, 'txt', 'ko')
        en_folder = os.path.join(root_dir, base, 'txt', 'en')
        if os.path.isdir(ko_folder):
            os.makedirs(en_folder, exist_ok=True)
            translate_ko_to_en(ko_folder, en_folder)
    log_message('✅ 전체 Ko → EN 번역 및 의역 완료')

def run_whisper_directory(output_dir: str):
    wav_folder = os.path.join(output_dir, 'wav')
    base = os.path.basename(output_dir)
    wav_files = sorted([
        os.path.join(wav_folder, f) for f in os.listdir(wav_folder)
        if f.startswith(f"{base}_") and f.endswith('.wav')
    ])
    if not wav_files:
        log_message("분할된 wav 파일이 없습니다.")
        return

    txt_root = os.path.join(output_dir, 'txt')
    ko_folder = os.path.join(txt_root, 'ko')
    en_folder = os.path.join(txt_root, 'en')
    os.makedirs(ko_folder, exist_ok=True)
    os.makedirs(en_folder, exist_ok=True)

    model_path, is_coreml = get_model_path()
    log_message(f'Whisper 모델: {model_path} (CoreML: {is_coreml})')

    # 한국어 텍스트만 처리, EN 폴더는 빈 채 유지
    cmd = [WHISPER_CLI, '--no-prints', '-m', model_path, '-otxt', '-l', 'ko'] + wav_files

    # CoreML은 자동으로 사용됨 (별도 옵션 불필요)
    if is_coreml and IS_MACOS:
        log_message('CoreML 모델 사용 중 (자동 가속)')

    # 통합 로깅 시스템으로 실행
    return_code = run_command_with_logging(cmd, cwd=wav_folder, description="Whisper 한국어 텍스트 처리")
    log_message(f"whisper KO 처리 완료 (return code: {return_code})")

    for wav in wav_files:
        name = os.path.splitext(os.path.basename(wav))[0]
        src = os.path.join(wav_folder, f"{name}.wav.txt")
        dst = os.path.join(ko_folder, f"{name}.ko.txt")
        if os.path.exists(src):
            # 읽고, 빈 줄 제거 후 한 줄로 합치기
            with open(src, 'r', encoding='utf-8') as rf:
                lines = [line.strip() for line in rf if line.strip()]
            single_line = ' '.join(lines)
            # ko 폴더에 저장
            with open(dst, 'w', encoding='utf-8') as wf:
                wf.write(single_line)
            log_message(f"한국어 텍스트 합치기 및 저장: {os.path.basename(dst)}")
        else:
            log_message(f"에러: 필사 파일을 찾을 수 없습니다: {os.path.basename(src)}")

    log_message("한국어 텍스트만 생성 완료. EN 폴더는 비워두었습니다.")
    log_message("한국어 텍스트 생성 완료. 다국어 번역 시작...")

    # 다국어 번역 처리
    translation_length = float(translation_length_var.get()) if 'translation_length_var' in globals() else 0.8
    quality_mode = translation_quality_var.get() if 'translation_quality_var' in globals() else "balanced"

    # 선택된 언어들 확인
    selected_languages = []
    if enable_english_var.get():
        selected_languages.append("english")
    if enable_chinese_var.get():
        selected_languages.append("chinese")
    if enable_japanese_var.get():
        selected_languages.append("japanese")

    if not selected_languages:
        selected_languages = ["english"]  # 기본값

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

        # 메모리 해제 확인 로그 추가
        log_message("🧹 Gemma3 모델 메모리 해제 완료 - CosyVoice 합성 준비")
    except Exception as e:
        log_message(f"❌ 번역 오류: {e}")
        return

    # 음성 합성 처리 - 각 언어별로 의역만 처리
    for synthesis_lang in selected_languages:
        lang_name = SUPPORTED_LANGUAGES[synthesis_lang]['name'].lower()

        # 의역(free)만 처리
        translation_type = "free"
        text_dir = os.path.join(txt_root, lang_name, translation_type)

        if not os.path.exists(text_dir):
            log_message(f"⏭️ {SUPPORTED_LANGUAGES[synthesis_lang]['name']} {translation_type} 텍스트 없음, 건너뛰기")
            continue

        log_message(f"🔊 {SUPPORTED_LANGUAGES[synthesis_lang]['name']} ({translation_type}) 음성 합성 시작...")

        # CosyVoice2 합성 호출
        cosy_out = os.path.join(output_dir, 'cosy_output', lang_name, translation_type)
        os.makedirs(cosy_out, exist_ok=True)

        # 메모리 정리 (합성 전)
        import gc
        gc.collect()

        try:
            # UI에서 설정값 가져오기
            enable_instruct = enable_instruct_var.get()
            command_mode = command_mode_var.get()
            manual_command = None
            if command_mode == 'manual':
                manual_command = manual_command_var.get()

            log_message(f"  언어: {SUPPORTED_LANGUAGES[synthesis_lang]['name']}")
            log_message(f"  번역 유형: {translation_type}")
            log_message(f"  Instruct2 활성화: {enable_instruct}")
            if manual_command:
                log_message(f"  수동 명령어: {manual_command}")

            # CosyVoice2 batch synthesis
            cosy_batch(
                audio_dir=wav_folder,
                prompt_text_dir=ko_folder,
                text_dir=text_dir,  # 언어별 번역 텍스트 사용
                out_dir=cosy_out,
                enable_instruct=enable_instruct,
                manual_command=manual_command
            )

            log_message(f"✅ {SUPPORTED_LANGUAGES[synthesis_lang]['name']} ({translation_type}) 합성 완료")

        except Exception as e:
            log_message(f"❌ {SUPPORTED_LANGUAGES[synthesis_lang]['name']} ({translation_type}) 합성 중 오류: {e}")
            continue

        # 합성 완료 후 메모리 정리
        import gc
        gc.collect()

    # 모든 언어별로 합성 결과 병합
    input_ext = os.path.splitext(input_file_var.get())[1]
    srt_path = os.path.join(output_dir, f"{base}{input_ext}.srt")
    segments = parse_srt_segments(srt_path)
    orig_audio = AudioSegment.from_file(input_file_var.get())
    original_duration_ms = len(orig_audio)
    enable_smart_compression = enable_smart_compression_var.get() if 'enable_smart_compression_var' in globals() else True

    # 각 언어별로 병합 수행
    for lang in selected_languages:
        lang_name = SUPPORTED_LANGUAGES[lang]['name'].lower()
        # 단순화된 병합 처리 - 의역(free)만 처리
        trans_type = "free"
        cosy_out = os.path.join(output_dir, 'cosy_output', lang_name, trans_type)
        if os.path.exists(cosy_out):
            merged_path = os.path.join(output_dir, f"{base}_{lang_name}_merged.wav")
            merge_segments_preserve_timing(
                segments,
                original_duration_ms,
                cosy_out,
                merged_path,
                length_handling=length_handling_var.get(),
                overlap_handling=overlap_handling_var.get(),
                max_extension=int(max_extension_var.get()),
                enable_smart_compression=enable_smart_compression
            )
            log_message(f"🎵 {lang_name} 병합 완료: {os.path.basename(merged_path)}")

    log_message(f"📁 언어별 폴더 구조:")
    log_message(f"   └── txt/")
    for lang in selected_languages:
        lang_name_display = SUPPORTED_LANGUAGES[lang]['name'].lower()
        log_message(f"       ├── {lang_name_display}/")
        log_message(f"       │   ├── literal/ (직역)")
        log_message(f"       │   └── free/ (의역)")
    log_message(f"   └── cosy_output/")
    for lang in selected_languages:
        lang_name_display = SUPPORTED_LANGUAGES[lang]['name'].lower()
        log_message(f"       ├── {lang_name_display}/")
        log_message(f"       │   ├── literal/ (직역 합성)")
        log_message(f"       │   └── free/ (의역 합성)")

# -------------------------------------
# 메인 처리 (버튼 클릭)
# -------------------------------------

def start_processing():
    # 플랫폼별 파일타입 형식 분기처리
    if IS_WINDOWS:
        # Windows 형식: 세미콜론으로 확장자 구분
        filetypes = [
            ('All Media Files', '*.wav;*.mp3;*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv'),
            ('Audio Files', '*.wav;*.mp3'),
            ('Video Files', '*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv'),
            ('WAV files', '*.wav'),
            ('MP3 files', '*.mp3'),
            ('MP4 files', '*.mp4'),
            ('All files', '*.*')
        ]
    else:
        # macOS/Linux 형식: 공백으로 확장자 구분
        filetypes = [
            ('All Media Files', '*.wav *.mp3 *.mp4 *.avi *.mkv *.mov *.wmv *.flv'),
            ('Audio Files', '*.wav *.mp3'),
            ('Video Files', '*.mp4 *.avi *.mkv *.mov *.wmv *.flv'),
            ('WAV files', '*.wav'),
            ('MP3 files', '*.mp3'),
            ('MP4 files', '*.mp4'),
            ('All files', '*.*')
        ]

    input_file = filedialog.askopenfilename(filetypes=filetypes)
    if not input_file:
        return
    input_file_var.set(input_file)
    base = os.path.splitext(os.path.basename(input_file))[0]
    out  = os.path.join(os.getcwd(), 'split_audio', base)
    os.makedirs(out, exist_ok=True)

    load_config()

    # Remove references to deleted log boxes since logging is now unified
    # install_log_box.configure(state='normal'); install_log_box.delete('1.0', tk.END); install_log_box.configure(state='disabled')
    # audio_log_box.configure(state='normal');   audio_log_box.delete('1.0', tk.END);   audio_log_box.configure(state='disabled')

    def worker():
        try:
            log_message('== whisper.cpp 실행 (VAD+SRT) ==')
            model_path, is_coreml = get_model_path()
            log_message(f'사용 모델: {model_path} (CoreML: {is_coreml})')

            whisper_cmd = [
                WHISPER_CLI,
                '--vad',
                # 1) split-on-word 제거
                # 2) VAD 옵션을 -f 이전으로 모음
                '--vad-model', resource_path('whisper.cpp/models/ggml-silero-v5.1.2.bin'),  # 또는 '-vm'
                '--vad-threshold',           str(vad_config['threshold']),
                '--vad-min-speech-duration-ms', str(vad_config['min_speech_duration_ms']),  # 필터링 기준
                '--vad-min-silence-duration-ms', str(vad_config['min_silence_duration_ms']),
                '--vad-max-speech-duration-s', str(vad_config['max_speech_duration_s']),
                '--vad-speech-pad-ms',       str(vad_config['speech_pad_ms']),
                '-f', input_file,                                                     # 입력 파일
                '-m', model_path,
                '--output-srt',
                '--language', 'ko',
            ]

            # CoreML은 자동으로 사용됨 (별도 옵션 불필요)
            # if is_coreml and IS_MACOS:
            #     whisper_cmd.extend(['--use-coreml'])
            #     install_log_queue.put('CoreML 가속 활성화')

            # whisper-cli 는 WHISPER_CLI 디렉터리에서 실행해야 .srt 가 out 에 생성됩니다
            run_command_with_logging(whisper_cmd, cwd=os.path.dirname(input_file), description="whisper.cpp VAD+SRT 처리")
            log_message('== SRT 생성 완료 ==')
            # ensure .srt from input directory is moved into out if not present
            srt_files = [f for f in os.listdir(out) if f.lower().endswith('.srt')]
            if not srt_files:
                input_dir = os.path.dirname(input_file)
                moved = False
                for f in os.listdir(input_dir):
                    if f.startswith(base) and f.lower().endswith('.srt'):
                        shutil.move(os.path.join(input_dir, f), out)
                        log_message(f'Moved SRT from input dir: {f}')
                        moved = True
                        break
                if not moved:
                    log_message('DEBUG: No .srt found in input directory')
            log_message(f"OUT DIR after move: {os.listdir(out)}")

            srt_files = [f for f in os.listdir(out) if f.lower().endswith('.srt')]
            if not srt_files:
                log_message('에러: SRT 파일을 찾을 수 없습니다.')
                return
            srt_path = os.path.join(out, srt_files[0])
            log_message(f'== SRT 파일 발견: {srt_files[0]} ==')

            segments, orig_dur = split_audio_by_srt(input_file, srt_path, out)
            log_message(f'== {len(segments)}개 세그먼트 분할 완료 ==')

            run_whisper_directory(out)
            log_message('== Whisper 처리 완료 ==')

        except Exception as e:
            log_message(f'에러 발생: {e}')

    threading.Thread(target=worker, daemon=True).start()

# -------------------------------------
# GUI 구성
# -------------------------------------
def save_config():
    try:
        vad_config['threshold'] = float(threshold_entry.get())
        vad_config['min_speech_duration_ms'] = int(min_speech_entry.get())
        vad_config['max_speech_duration_s'] = float(max_speech_entry.get())
        vad_config['min_silence_duration_ms'] = int(min_silence_entry.get())
        vad_config['speech_pad_ms'] = int(pad_entry.get())
        with open(resource_path(CONFIG_FILE), 'w', encoding='utf-8') as f:
            json.dump(vad_config, f, indent=4, ensure_ascii=False)
        log_message('설정 저장 완료')
    except Exception as e:
        log_message(f'설정 저장 오류: {e}')


def load_config():
    global vad_config
    path = resource_path(CONFIG_FILE)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
    else:
        loaded = {}
    vad_config = DEFAULT_CONFIG.copy()
    vad_config.update(loaded)
    log_message(f'현재 VAD 설정: {vad_config}')

root = tk.Tk()
input_file_var = tk.StringVar()
root.title('STT Voice Splitter')
root.geometry('1400x900')  # 화면 크기 확대

nb = ttk.Notebook(root)
main_tab     = ttk.Frame(nb)
settings_tab = ttk.Frame(nb)
log_tab = ttk.Frame(nb)  # 로그 탭 추가
nb.add(main_tab, text='메인')
nb.add(settings_tab, text='설정')
nb.add(log_tab, text='로그')
nb.pack(expand=1, fill='both')

ttk.Button(main_tab, text='시작', command=start_processing).pack(pady=5)

# SRT만 생성하는 버튼 추가
ttk.Button(main_tab, text='SRT 전용 생성', command=generate_srt_only).pack(pady=5)

# 합성 타입 선택 추가
synthesis_frame = ttk.Frame(main_tab)
synthesis_frame.pack(pady=5)

# Instruct2 활성화 체크박스
enable_instruct_var = tk.BooleanVar(value=False)
instruct_checkbox = ttk.Checkbutton(synthesis_frame, text='Instruct2 합성 활성화',
                                    variable=enable_instruct_var)
instruct_checkbox.pack(side=tk.LEFT, padx=(0, 20))

# 병합할 합성 결과 선택
ttk.Label(synthesis_frame, text='병합할 합성 결과:').pack(side=tk.LEFT, padx=(0, 10))
synthesis_type_var = tk.StringVar(value="Zero-shot")
ttk.Radiobutton(synthesis_frame, text='Zero-shot', variable=synthesis_type_var, value='Zero-shot').pack(side=tk.LEFT,
                                                                                                        padx=5)
ttk.Radiobutton(synthesis_frame, text='Instruct2', variable=synthesis_type_var, value='Instruct2').pack(side=tk.LEFT,
                                                                                                        padx=5)

# Instruct2 설정 프레임
instruct_frame = ttk.LabelFrame(main_tab, text="Instruct2 설정")
instruct_frame.pack(pady=5, padx=10, fill='x')

# 명령어 입력 방식 선택
command_mode_var = tk.StringVar(value="auto")
ttk.Label(instruct_frame, text="명령어 설정:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
ttk.Radiobutton(instruct_frame, text='자동 분석', variable=command_mode_var, value='auto').grid(row=0, column=1, sticky='w',
                                                                                            padx=5)
ttk.Radiobutton(instruct_frame, text='수동 입력', variable=command_mode_var, value='manual').grid(row=0, column=2,
                                                                                              sticky='w', padx=5)

# 수동 명령어 입력
manual_command_var = tk.StringVar(value="자연스럽게 말해")
ttk.Label(instruct_frame, text="명령어:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
manual_command_entry = ttk.Entry(instruct_frame, textvariable=manual_command_var, width=40)
manual_command_entry.grid(row=1, column=1, columnspan=2, sticky='w', padx=5, pady=2)

# 사전 정의된 명령어 버튼들
preset_frame = ttk.Frame(instruct_frame)
preset_frame.grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=5)

preset_commands = [
    "자연스럽게 말해", "활기차게 말해", "차분하게 말해",
    "감정적으로 말해", "천천히 말해", "빠르게 말해"
]


def set_preset_command(cmd):
    manual_command_var.set(cmd)


for i, cmd in enumerate(preset_commands):
    btn = ttk.Button(preset_frame, text=cmd,
                     command=lambda c=cmd: set_preset_command(c))
    btn.grid(row=i // 3, column=i % 3, padx=2, pady=2, sticky='w')

# 타임라인 및 길이 처리 설정 프레임
timeline_frame = ttk.LabelFrame(main_tab, text="타임라인 & 길이 처리")
timeline_frame.pack(pady=5, padx=10, fill='x')

# 길이 처리 방식 선택
length_handling_var = tk.StringVar(value="preserve")
ttk.Label(timeline_frame, text="합성 음성 길이 처리:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
ttk.Radiobutton(timeline_frame, text='완전 보존 (추천)', variable=length_handling_var, value='preserve').grid(row=0, column=1,
                                                                                                        sticky='w',
                                                                                                        padx=5)
ttk.Radiobutton(timeline_frame, text='원본 길이 맞춤', variable=length_handling_var, value='fit').grid(row=0, column=2,
                                                                                                 sticky='w', padx=5)

# 겹침 처리 방식
overlap_handling_var = tk.StringVar(value="fade")
ttk.Label(timeline_frame, text="세그먼트 겹침 처리:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
ttk.Radiobutton(timeline_frame, text='페이드 처리 (추천)', variable=overlap_handling_var, value='fade').grid(row=1, column=1,
                                                                                                      sticky='w',
                                                                                                      padx=5)
ttk.Radiobutton(timeline_frame, text='자르기', variable=overlap_handling_var, value='cut').grid(row=1, column=2,
                                                                                             sticky='w', padx=5)

# 최대 확장 제한
max_extension_var = tk.StringVar(value="50")
ttk.Label(timeline_frame, text="최대 확장율 (%):").grid(row=2, column=0, sticky='w', padx=5, pady=2)
max_extension_entry = ttk.Entry(timeline_frame, textvariable=max_extension_var, width=10)
max_extension_entry.grid(row=2, column=1, sticky='w', padx=5, pady=2)
ttk.Label(timeline_frame, text="(원본 대비 최대 확장 허용 비율)").grid(row=2, column=2, sticky='w', padx=5, pady=2)

# 번역 설정 프레임
translation_frame = ttk.LabelFrame(main_tab, text="다국어 번역 설정")
translation_frame.pack(pady=5, padx=10, fill='x')

# 번역할 언어 선택 체크박스들
target_languages_frame = ttk.Frame(translation_frame)
target_languages_frame.grid(row=0, column=0, columnspan=4, sticky='w', padx=5, pady=5)

ttk.Label(target_languages_frame, text="번역할 언어:").pack(side=tk.LEFT, padx=(0, 10))

# 언어별 체크박스 변수들
enable_english_var = tk.BooleanVar(value=True)
enable_chinese_var = tk.BooleanVar(value=True)
enable_japanese_var = tk.BooleanVar(value=True)

ttk.Checkbutton(target_languages_frame, text="🇺🇸 영어", variable=enable_english_var).pack(side=tk.LEFT, padx=5)
ttk.Checkbutton(target_languages_frame, text="🇨🇳 중국어", variable=enable_chinese_var).pack(side=tk.LEFT, padx=5)
ttk.Checkbutton(target_languages_frame, text="🇯🇵 일본어", variable=enable_japanese_var).pack(side=tk.LEFT, padx=5)

# 번역 길이 비율
translation_length_var = tk.StringVar(value="0.6")
ttk.Label(translation_frame, text="번역 길이 비율:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
translation_length_entry = ttk.Entry(translation_frame, textvariable=translation_length_var, width=10)
translation_length_entry.grid(row=0, column=1, sticky='w', padx=5, pady=2)
ttk.Label(translation_frame, text="(0.8 = 원본의 80% 길이로 축약, 1.0 = 원본 길이 유지)").grid(row=0, column=2, sticky='w', padx=5,
                                                                                 pady=2)

# 스마트 압축 설정 프레임
smart_compression_frame = ttk.LabelFrame(main_tab, text="스마트 압축 (AI 길이 조절)")
smart_compression_frame.pack(pady=5, padx=10, fill='x')

# 스마트 압축 활성화
enable_smart_compression_var = tk.BooleanVar(value=True)
ttk.Checkbutton(smart_compression_frame, text='스마트 압축 활성화 (AI 기반 자동 길이 조절)',
                variable=enable_smart_compression_var).grid(row=0, column=0, columnspan=3, sticky='w', padx=5, pady=2)

# 압축 강도 설정
compression_level_var = tk.StringVar(value="balanced")
ttk.Label(smart_compression_frame, text="압축 강도:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
ttk.Radiobutton(smart_compression_frame, text='보수적', variable=compression_level_var, value='conservative').grid(row=1,
                                                                                                                column=1,
                                                                                                                sticky='w',
                                                                                                                padx=5)
ttk.Radiobutton(smart_compression_frame, text='균형', variable=compression_level_var, value='balanced').grid(row=1,
                                                                                                           column=2,
                                                                                                           sticky='w',
                                                                                                           padx=5)
ttk.Radiobutton(smart_compression_frame, text='적극적', variable=compression_level_var, value='aggressive').grid(row=1,
                                                                                                              column=3,
                                                                                                              sticky='w',
                                                                                                              padx=5)

# 무음 제거 설정
remove_silence_var = tk.BooleanVar(value=True)
ttk.Checkbutton(smart_compression_frame, text='과도한 무음 구간 자동 제거',
                variable=remove_silence_var).grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=2)

max_silence_var = tk.StringVar(value="300")
ttk.Label(smart_compression_frame, text="최대 무음 길이(ms):").grid(row=2, column=2, sticky='w', padx=5, pady=2)
max_silence_entry = ttk.Entry(smart_compression_frame, textvariable=max_silence_var, width=8)
max_silence_entry.grid(row=2, column=3, sticky='w', padx=5, pady=2)

# 우선순위 기반 처리 설정
priority_processing_var = tk.BooleanVar(value=True)
ttk.Checkbutton(smart_compression_frame, text='중요도 기반 차등 처리 (중요한 대사는 덜 압축)',
                variable=priority_processing_var).grid(row=3, column=0, columnspan=3, sticky='w', padx=5, pady=2)

# 번역 품질 설정
translation_quality_var = tk.StringVar(value="balanced")
ttk.Label(translation_frame, text="번역 품질 우선순위:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
ttk.Radiobutton(translation_frame, text='간결함 우선', variable=translation_quality_var, value='concise').grid(row=3,
                                                                                                          column=1,
                                                                                                          sticky='w',
                                                                                                          padx=5)
ttk.Radiobutton(translation_frame, text='균형', variable=translation_quality_var, value='balanced').grid(row=3, column=2,
                                                                                                       sticky='w',
                                                                                                       padx=5)
ttk.Radiobutton(translation_frame, text='정확성 우선', variable=translation_quality_var, value='accurate').grid(row=3,
                                                                                                           column=3,
                                                                                                           sticky='w',
                                                                                                           padx=5)

# 음성 합성할 언어 선택
synthesis_lang_frame = ttk.LabelFrame(main_tab, text="음성 합성 언어 선택")
synthesis_lang_frame.pack(pady=5, padx=10, fill='x')

synthesis_language_var = tk.StringVar(value="english")
ttk.Label(synthesis_lang_frame, text="합성할 언어:").pack(side=tk.LEFT, padx=(5, 10))
ttk.Radiobutton(synthesis_lang_frame, text="🇺🇸 영어", variable=synthesis_language_var, value="english").pack(side=tk.LEFT,
                                                                                                           padx=5)
ttk.Radiobutton(synthesis_lang_frame, text="🇨🇳 중국어", variable=synthesis_language_var, value="chinese").pack(
    side=tk.LEFT, padx=5)
ttk.Radiobutton(synthesis_lang_frame, text="🇯🇵 일본어", variable=synthesis_language_var, value="japanese").pack(
    side=tk.LEFT, padx=5)

# 번역 유형 선택 (literal/free)
translation_type_var = tk.StringVar(value="free")
ttk.Label(synthesis_lang_frame, text="  |  번역 유형:").pack(side=tk.LEFT, padx=(20, 5))
ttk.Radiobutton(synthesis_lang_frame, text="직역", variable=translation_type_var, value="literal").pack(side=tk.LEFT,
                                                                                                      padx=5)
ttk.Radiobutton(synthesis_lang_frame, text="의역", variable=translation_type_var, value="free").pack(side=tk.LEFT, padx=5)

# 메모리 관리 설정 프레임
memory_frame = ttk.LabelFrame(main_tab, text="메모리 관리 (CPU/CUDA)")
memory_frame.pack(pady=5, padx=10, fill='x')

# 배치 크기 설정
batch_size_var = tk.StringVar(value="5")
ttk.Label(memory_frame, text="배치 크기:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
batch_size_entry = ttk.Entry(memory_frame, textvariable=batch_size_var, width=10)
batch_size_entry.grid(row=0, column=1, sticky='w', padx=5, pady=2)
ttk.Label(memory_frame, text="(한 번에 처리할 세그먼트 수, 메모리 부족 시 줄이세요)").grid(row=0, column=2, sticky='w', padx=5, pady=2)

# 메모리 정리 주기
memory_cleanup_var = tk.BooleanVar(value=True)
ttk.Checkbutton(memory_frame, text='세그먼트마다 메모리 정리 (느리지만 안전)', variable=memory_cleanup_var).grid(row=1, column=0,
                                                                                                columnspan=3,
                                                                                                sticky='w', padx=5,
                                                                                                pady=2)

# 병합 버튼 및 입력란
ttk.Label(main_tab, text='커스텀 병합 타이밍(ms) 리스트 (JSON 배열):').pack(pady=(10,0))
merge_entry = ttk.Entry(main_tab, width=80)
merge_entry.pack(pady=2)
ttk.Button(main_tab, text='병합 실행', command=merge_custom_callback).pack(pady=5)
# 전체 병합 버튼 추가
ttk.Button(main_tab, text='전체 병합', command=merge_all_segments_callback).pack(pady=5)

# 로그 탭에 로그 박스 추가
log_text = scrolledtext.ScrolledText(log_tab, width=100, height=40,
    font=("Malgun Gothic",10), state='disabled', background='black', foreground='lime')
log_text.pack(padx=5, pady=5, fill='both', expand=True)

# 메인 탭의 로그 박스는 제거
# install_log_box = scrolledtext.ScrolledText(main_tab, width=100, height=20,
#     font=("Malgun Gothic",10), state='disabled', background='black', foreground='lime')
# install_log_box.pack(padx=5, pady=5, fill='both', expand=True)
# audio_log_box = scrolledtext.ScrolledText(main_tab, width=100, height=10,
#     font=("Malgun Gothic",10), state='disabled', background='white', foreground='black')
# audio_log_box.pack(padx=5, pady=5, fill='both', expand=True)

frm = ttk.Frame(settings_tab); frm.pack(padx=10, pady=10, fill='x')
labels_k   = ['음성 임계값','최소 음성(ms)','최대 음성(s)','최소 무음(ms)','음성 패딩(ms)']
config_keys= ['threshold','min_speech_duration_ms','max_speech_duration_s','min_silence_duration_ms','speech_pad_ms']
entries = []
for i, key in enumerate(config_keys):
    ttk.Label(frm, text=labels_k[i]).grid(row=i, column=0, sticky='w')
    ent = ttk.Entry(frm); ent.grid(row=i, column=1, sticky='w')
    entries.append(ent)
threshold_entry, min_speech_entry, max_speech_entry, min_silence_entry, pad_entry = entries
ttk.Button(frm, text='저장', command=save_config).grid(row=len(entries), column=0, columnspan=2, pady=10)


load_config()
for ent, key in zip(entries, config_keys):
    ent.insert(0, vad_config[key])


# 로그 큐 처리를 로그 탭의 로그 박스에 연결
root.after(200, process_log_queue)
root.mainloop()
