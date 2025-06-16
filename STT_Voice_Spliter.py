import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading, platform, os, subprocess, queue, json, sys, shutil, re
import sys
import os
# CosyVoice 패키지 및 의존 모듈을 찾을 수 있도록 경로 추가
repo_root = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(repo_root, 'CosyVoice'))
sys.path.insert(0, os.path.join(repo_root, 'CosyVoice', 'third_party'))
# sys.path.insert(0, os.path.join(repo_root, 'CosyVoice', 'third_party', 'matcha'))
sys.path.insert(0, os.path.join(repo_root, 'CosyVoice', 'third_party', 'Matcha-TTS'))
from batch_cosy import main as cosy_batch
from pydub import AudioSegment  # pip install pydub
from batch_translate import batch_translate

def merge_custom_callback():
    try:
        # load original segments and duration
        base = os.path.splitext(os.path.basename(input_file_var.get()))[0]
        out_dir = os.path.join(os.getcwd(), 'split_audio', base)
        srt_path = os.path.join(out_dir, f"{base}.wav.srt")
        segments = parse_srt_segments(srt_path)
        # load original full audio directly from the input path
        input_path = input_file_var.get()
        orig_audio = AudioSegment.from_file(input_path)
        orig_dur = len(orig_audio)
        # parse user timings list
        timings = json.loads(merge_entry.get())
        # merge using only specified segments indices
        selected = [segments[i-1] for i in timings]
        merged_path = os.path.join(out_dir, f"{base}_merged_custom.wav")
        merge_segments_preserve_timing(selected, orig_dur, os.path.join(out_dir, 'wav'), merged_path)
    except Exception as e:
        install_log_queue.put(f"병합 오류: {e}")

# 전체 세그먼트 병합 콜백 함수
def merge_all_segments_callback():
    try:
        base = os.path.splitext(os.path.basename(input_file_var.get()))[0]
        out_dir = os.path.join(os.getcwd(), 'split_audio', base)
        srt_path = os.path.join(out_dir, f"{base}.wav.srt")
        segments = parse_srt_segments(srt_path)
        input_path = input_file_var.get()
        orig_audio = AudioSegment.from_file(input_path)
        original_duration_ms = len(orig_audio)
        merged_path = os.path.join(out_dir, f"{base}_merged_all.wav")
        merge_segments_preserve_timing(segments, original_duration_ms, os.path.join(out_dir, 'wav'), merged_path)
    except Exception as e:
        install_log_queue.put(f"전체 병합 오류: {e}")

# ------------------------
# OS/환경 정보 및 경로 처리
# ------------------------
IS_WINDOWS = platform.system() == "Windows"

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.getcwd()
    if IS_WINDOWS and relative_path.endswith('whisper-cli'):
        relative_path += '.exe'
    return os.path.join(base_path, relative_path)

# whisper-cli 경로
if IS_WINDOWS:
    WHISPER_CLI = resource_path('whisper.cpp/build/bin/Release/whisper-cli')
    if not os.path.exists(WHISPER_CLI):
        WHISPER_CLI = resource_path('whisper.cpp/build/bin/whisper-cli')
else:
    WHISPER_CLI = resource_path('whisper.cpp/build/bin/whisper-cli')

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
    'speech_pad_ms': 200,
}

install_log_queue = queue.Queue()
audio_log_queue   = queue.Queue()
vad_config        = {}

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
    install_log_queue.put(f'현재 VAD 설정: {vad_config}')

def save_config():
    try:
        vad_config['threshold']               = float(threshold_entry.get())
        vad_config['min_speech_duration_ms']  = int(min_speech_entry.get())
        vad_config['max_speech_duration_s']   = float(max_speech_entry.get())
        vad_config['min_silence_duration_ms'] = int(min_silence_entry.get())
        vad_config['speech_pad_ms']           = int(pad_entry.get())
        with open(resource_path(CONFIG_FILE), 'w', encoding='utf-8') as f:
            json.dump(vad_config, f, indent=4, ensure_ascii=False)
        settings_log_box.configure(state='normal')
        settings_log_box.insert(tk.END, '설정 저장 완료\n')
        settings_log_box.configure(state='disabled')
    except Exception as e:
        settings_log_box.configure(state='normal')
        settings_log_box.insert(tk.END, f'설정 저장 오류: {e}\n')
        settings_log_box.configure(state='disabled')

# -------------------------------------
# 로그 처리
# -------------------------------------
def process_install_log_queue():
    while not install_log_queue.empty():
        line = install_log_queue.get()
        install_log_box.configure(state='normal')
        install_log_box.insert(tk.END, line + '\n')
        install_log_box.see(tk.END)
        install_log_box.configure(state='disabled')
    root.after(200, process_install_log_queue)

def process_audio_log_queue():
    while not audio_log_queue.empty():
        line = audio_log_queue.get()
        audio_log_box.configure(state='normal')
        audio_log_box.insert(tk.END, line + '\n')
        audio_log_box.see(tk.END)
        audio_log_box.configure(state='disabled')
    root.after(200, process_audio_log_queue)

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
    install_log_queue.put(f'DEBUG: SRT 경로 → {srt_path} / 존재 → {os.path.exists(srt_path)}')
    segments = parse_srt_segments(srt_path)
    install_log_queue.put(f'DEBUG: 파싱된 세그먼트 수 → {len(segments)}')

    audio = AudioSegment.from_file(audio_path)
    wav_folder = os.path.join(output_dir, 'wav')
    base = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(wav_folder, exist_ok=True)

    for idx, (start_ms, end_ms) in enumerate(segments, 1):
        chunk = audio[start_ms:end_ms]
        out_path = os.path.join(wav_folder, f"{base}_{idx:03d}.wav")
        chunk.export(out_path, format="wav")
        audio_log_queue.put(f"세그먼트 {idx}: {start_ms}~{end_ms}")
    return segments, len(audio)

# -------------------------------------
# 세그먼트 병합 (원본 타이밍 보존)
# -------------------------------------
def merge_segments_preserve_timing(segments, original_duration_ms, segments_dir, output_path):
    # derive base name from directory
    base = os.path.basename(os.path.dirname(segments_dir))
    merged = AudioSegment.silent(duration=0)
    for idx, (start_ms, end_ms) in enumerate(segments, start=1):
        # 시작 지연을 위해 무음 삽입
        if len(merged) < start_ms:
            merged += AudioSegment.silent(duration=start_ms - len(merged))
        seg_path = os.path.join(segments_dir, f"{base}_{idx:03d}.wav")
        merged += AudioSegment.from_file(seg_path)
    # 부족한 부분 채우기
    if len(merged) < original_duration_ms:
        merged += AudioSegment.silent(duration=int(original_duration_ms - len(merged)))
    merged.export(output_path, format="wav")
    install_log_queue.put(f"병합 완료: {output_path}")

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
                install_log_queue.put(f"번역 완료: {fname} → {os.path.basename(en_path)}")
            except Exception as e:
                install_log_queue.put(f"번역 오류: {fname} ({e})")
    install_log_queue.put(f"✅ {ko_folder} 내 Ko → EN 번역 및 의역 완료")

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
        install_log_queue.put(f"경로가 없습니다: {root_dir}")
        return

    for base in os.listdir(root_dir):
        ko_folder = os.path.join(root_dir, base, 'txt', 'ko')
        en_folder = os.path.join(root_dir, base, 'txt', 'en')
        if os.path.isdir(ko_folder):
            os.makedirs(en_folder, exist_ok=True)
            translate_ko_to_en(ko_folder, en_folder)
    install_log_queue.put('✅ 전체 Ko → EN 번역 및 의역 완료')

def run_whisper_directory(output_dir: str):
    wav_folder = os.path.join(output_dir, 'wav')
    base = os.path.basename(output_dir)
    wav_files = sorted([
        os.path.join(wav_folder, f) for f in os.listdir(wav_folder)
        if f.startswith(f"{base}_") and f.endswith('.wav')
    ])
    if not wav_files:
        install_log_queue.put("분할된 wav 파일이 없습니다.")
        return

    txt_root = os.path.join(output_dir, 'txt')
    ko_folder = os.path.join(txt_root, 'ko')
    en_folder = os.path.join(txt_root, 'en')
    os.makedirs(ko_folder, exist_ok=True)
    os.makedirs(en_folder, exist_ok=True)

    MODEL = resource_path('resources/ggml-large-v3-turbo.bin')

    # 한국어 텍스트만 처리, EN 폴더는 빈 채 유지
    cmd = [WHISPER_CLI, '--no-prints', '-m', MODEL, '-otxt', '-l', 'ko'] + wav_files
    result = subprocess.run(cmd, cwd=wav_folder,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            encoding='utf-8', errors='ignore')
    install_log_queue.put(f"whisper KO returncode={result.returncode}")
    if result.stderr:
        install_log_queue.put(f"whisper KO stderr: {result.stderr}")

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
            install_log_queue.put(f"한국어 텍스트 합치기 및 저장: {os.path.basename(dst)}")
        else:
            install_log_queue.put(f"에러: 필사 파일을 찾을 수 없습니다: {os.path.basename(src)}")

    install_log_queue.put("한국어 텍스트만 생성 완료. EN 폴더는 비워두었습니다.")
    # Whisper 필사 완료 후 전체 번역 자동 실행
    batch_translate(ko_folder, en_folder)
    # free 폴더의 비어 있는 파일을 literal 폴더의 내용으로 대체
    free_dir = os.path.join(en_folder, 'free')
    lit_dir = os.path.join(en_folder, 'literal')
    for fname in os.listdir(free_dir):
        free_path = os.path.join(free_dir, fname)
        try:
            with open(free_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if not content:
                lit_path = os.path.join(lit_dir, fname)
                if os.path.exists(lit_path):
                    with open(lit_path, 'r', encoding='utf-8') as lf:
                        lit_text = lf.read()
                    with open(free_path, 'w', encoding='utf-8') as wf:
                        wf.write(lit_text)
                    install_log_queue.put(f"FREE 번역 비어있음 → LITERAL로 대체: {fname}")
                else:
                    install_log_queue.put(f"경고: FREEDIR과 LITDIR 모두에 파일이 없습니다: {fname}")
        except Exception as e:
            install_log_queue.put(f"Fallback 처리 오류: {fname} ({e})")
    # Gemma3 번역/의역 후 CosyVoice2 합성 호출
    cosy_out = os.path.join(output_dir, 'cosy_output')
    os.makedirs(cosy_out, exist_ok=True)
    install_log_queue.put("🔊 CosyVoice2 합성을 시작합니다...")
    try:
        # CosyVoice2 batch synthesis
        cosy_batch(
            audio_dir=wav_folder,
            prompt_text_dir=ko_folder,
            text_dir=os.path.join(en_folder, 'free'),
            out_dir=cosy_out
        )
    except Exception as e:
        install_log_queue.put(f"❌ CosyVoice2 합성 중 오류 발생: {e}")
    else:
        install_log_queue.put(f"✅ CosyVoice2 합성 완료: {cosy_out}")

    srt_path = os.path.join(output_dir, f"{base}.wav.srt")
    segments = parse_srt_segments(srt_path)

    # 2) 원본 오디오 전체 길이(ms) 구하기
    orig_path = input_file_var.get()  # 전역 변수로 선택된 입력 파일 경로
    orig_audio = AudioSegment.from_file(orig_path)
    original_duration_ms = len(orig_audio)

    # 3) 병합 실행
    merged_path = os.path.join(output_dir, f"{base}_cosy_merged.wav")
    merge_segments_preserve_timing(
        segments,
        original_duration_ms,
        cosy_out,      # CosyVoice2가 저장한 세그먼트 폴더
        merged_path
    )

# -------------------------------------
# 메인 처리 (버튼 클릭)
# -------------------------------------

def start_processing():
    input_file = filedialog.askopenfilename(filetypes=[('Audio Files','*.wav;*.mp3')])
    if not input_file:
        return
    input_file_var.set(input_file)
    base = os.path.splitext(os.path.basename(input_file))[0]
    out  = os.path.join(os.getcwd(), 'split_audio', base)
    os.makedirs(out, exist_ok=True)

    load_config()
    install_log_box.configure(state='normal'); install_log_box.delete('1.0', tk.END); install_log_box.configure(state='disabled')
    audio_log_box.configure(state='normal');   audio_log_box.delete('1.0', tk.END);   audio_log_box.configure(state='disabled')

    def worker():
        try:
            install_log_queue.put('== whisper.cpp 실행 (VAD+SRT) ==')
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
                '-m', resource_path('resources/ggml-large-v3-turbo.bin'),
                '--output-srt',
                '--language', 'ko',
            ]
            # whisper-cli 는 WHISPER_CLI 디렉터리에서 실행해야 .srt 가 out 에 생성됩니다
            subprocess.run(whisper_cmd, cwd=os.path.dirname(input_file), check=True)
            install_log_queue.put('== SRT 생성 완료 ==')
            # ensure .srt from input directory is moved into out if not present
            srt_files = [f for f in os.listdir(out) if f.lower().endswith('.srt')]
            if not srt_files:
                input_dir = os.path.dirname(input_file)
                moved = False
                for f in os.listdir(input_dir):
                    if f.startswith(base) and f.lower().endswith('.srt'):
                        shutil.move(os.path.join(input_dir, f), out)
                        install_log_queue.put(f'Moved SRT from input dir: {f}')
                        moved = True
                        break
                if not moved:
                    install_log_queue.put('DEBUG: No .srt found in input directory')
            install_log_queue.put(f"OUT DIR after move: {os.listdir(out)}")

            srt_files = [f for f in os.listdir(out) if f.lower().endswith('.srt')]
            if not srt_files:
                install_log_queue.put('에러: SRT 파일을 찾을 수 없습니다.')
                return
            srt_path = os.path.join(out, srt_files[0])
            install_log_queue.put(f'== SRT 파일 발견: {srt_files[0]} ==')

            segments, orig_dur = split_audio_by_srt(input_file, srt_path, out)
            install_log_queue.put(f'== {len(segments)}개 세그먼트 분할 완료 ==')

            run_whisper_directory(out)
            install_log_queue.put('== Whisper 처리 완료 ==')

        except Exception as e:
            install_log_queue.put(f'에러 발생: {e}')

    threading.Thread(target=worker, daemon=True).start()

# -------------------------------------
# GUI 구성
# -------------------------------------
root = tk.Tk()
input_file_var = tk.StringVar()
root.title('STT Voice Splitter')
root.geometry('900x700')

nb = ttk.Notebook(root)
main_tab     = ttk.Frame(nb)
settings_tab = ttk.Frame(nb)
nb.add(main_tab, text='메인')
nb.add(settings_tab, text='설정')
nb.pack(expand=1, fill='both')

ttk.Button(main_tab, text='시작', command=start_processing).pack(pady=5)
# 병합 버튼 및 입력란
ttk.Label(main_tab, text='커스텀 병합 타이밍(ms) 리스트 (JSON 배열):').pack(pady=(10,0))
merge_entry = ttk.Entry(main_tab, width=80)
merge_entry.pack(pady=2)
ttk.Button(main_tab, text='병합 실행', command=merge_custom_callback).pack(pady=5)
# 전체 병합 버튼 추가
ttk.Button(main_tab, text='전체 병합', command=merge_all_segments_callback).pack(pady=5)

install_log_box = scrolledtext.ScrolledText(main_tab, width=100, height=20,
    font=("Malgun Gothic",10), state='disabled', background='black', foreground='lime')
install_log_box.pack(padx=5, pady=5, fill='both', expand=True)
audio_log_box = scrolledtext.ScrolledText(main_tab, width=100, height=10,
    font=("Malgun Gothic",10), state='disabled', background='white', foreground='black')
audio_log_box.pack(padx=5, pady=5, fill='both', expand=True)

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

settings_log_box = scrolledtext.ScrolledText(settings_tab, width=80, height=8,
    font=("Malgun Gothic",10), state='disabled', background='white', foreground='black')
settings_log_box.pack(padx=5, pady=5, fill='both', expand=True)

load_config()
for ent, key in zip(entries, config_keys):
    ent.insert(0, vad_config[key])

root.after(200, process_install_log_queue)
root.after(200, process_audio_log_queue)
root.mainloop()