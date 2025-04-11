import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading, platform, os, subprocess, time, zipfile, shutil, queue, asyncio, json
import torch, torchaudio
import requests  # 파일 크기 확인용
import aiohttp, aiofiles  # 비동기 다운로드용
import sys  # sys.exit() 사용

# -------------------------------------
# PyInstaller 번들 리소스 경로 추출 함수
# -------------------------------------
def resource_path(relative_path):
    """
    PyInstaller 번들링된 실행 파일 내에서 리소스 파일의 절대 경로를 반환합니다.
    번들된 경우 sys._MEIPASS가 기본 경로로 설정되며, 그렇지 않으면 현재 작업 디렉토리를 사용합니다.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
    
# 외부 바이너리 경로 (PyInstaller 번들을 고려)
# ffmpeg_path = resource_path("ffmpeg")
# git_path    = resource_path("git")
# cmake_path  = resource_path("cmake")

ffmpeg_path = "ffmpeg"
git_path    = "git"
cmake_path  = "cmake"

# 사용 예시: ffmpeg 실행 (환경변수 PATH를 별도로 설정하여 번들 내 바이너리 우선 사용)
env = os.environ.copy()
env["PATH"] = f"{resource_path('.')}{os.pathsep}" + env.get("PATH", "")
command = f'"{ffmpeg_path}" -version'
os.system(command)

# -------------------------------------
# 시스템 의존성(홈브류를 통한 설치) 체크 및 자동 설치 함수
# -------------------------------------
def is_installed(command):
    """시스템 PATH에 해당 명령어가 존재하는지 확인"""
    return shutil.which(command) is not None

def install_with_brew(package):
    """Homebrew를 이용해 해당 패키지 설치"""
    try:
        print(f"{package} 설치를 시작합니다...")
        subprocess.run(["brew", "install", package], check=True)
        print(f"{package} 설치 완료.")
    except subprocess.CalledProcessError as e:
        print(f"{package} 설치 중 오류 발생: {e}")
        messagebox.showerror("의존성 설치 오류", f"{package} 설치 중 오류 발생:\n{e}")
        sys.exit(1)

def check_and_install_system_dependencies(progress_callback):
    """Homebrew 및 필수 시스템 의존성(예: git, ffmpeg, cmake)을 체크하고 없으면 설치"""
    if not is_installed("brew"):
        progress_callback("Homebrew가 설치되어 있지 않습니다.", 0)
        messagebox.showerror(
            "의존성 오류", 
            "Homebrew가 설치되어 있지 않습니다. 먼저 아래 명령어로 Homebrew를 설치하세요:\n"
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        )
        sys.exit(1)
    dependencies = [
        {"dep": "git", "brew": "git"},
        {"dep": "ffmpeg", "brew": "ffmpeg"},
        {"dep": "cmake", "brew": "cmake"}
    ]
    for d in dependencies:
        if not is_installed(d["dep"]):
            progress_callback(f"{d['dep']}가 설치되어 있지 않습니다. 설치 진행 중...", 2)
            install_with_brew(d["brew"])
            progress_callback(f"{d['dep']} 설치 완료.", 5)
        else:
            progress_callback(f"{d['dep']}이(가) 이미 설치되어 있습니다.", 2)

# -------------------------------------
# 전역 변수 및 설정
# -------------------------------------
install_log_queue = queue.Queue()
audio_log_queue   = queue.Queue()

CONFIG_FILE    = "config.json"
DEFAULT_CONFIG = {
    "min_speech_duration_ms": 500,
    "min_silence_duration_ms": 700,
    "max_speech_duration_s": 18,
    "speech_pad_ms": 10,
    "threshold": 0.6
}
vad_config = {}

def load_config():
    global vad_config
    config_path = resource_path(CONFIG_FILE)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                vad_config = json.load(f)
        except Exception as e:
            print("설정 파일 로드 에러:", e)
            vad_config = DEFAULT_CONFIG.copy()
    else:
        vad_config = DEFAULT_CONFIG.copy()
    print("현재 VAD 설정:", vad_config)

def save_config():
    with open(resource_path(CONFIG_FILE), "w") as f:
        json.dump(vad_config, f, indent=4)

load_config()

# Whisper CLI 및 모델 경로 (PyInstaller 번들 시 resource_path 적용)
WHISPER_CLI   = resource_path("whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = resource_path("whisper.cpp/models/ggml-large-v3-turbo.bin")

# -------------------------------------
# Helper 함수: 파일 크기 확인 및 비동기 다운로드
# -------------------------------------
def get_file_size(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
    return total

async def download_with_progress_aiohttp(url, filename, progress_callback=None):
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/109.0.0.0 Safari/537.36"),
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 65536  # 64KB
            async with aiofiles.open(filename, "wb") as f:
                async for chunk in response.content.iter_chunked(chunk_size):
                    if chunk:
                        await f.write(chunk)
                        downloaded += len(chunk)
                        if total and progress_callback:
                            percent = downloaded * 100 / total
                            progress_callback(f"다운로드 진행률: {percent:.1f}%", min(100, percent))

# -------------------------------------
# 설치 관리자 관련 함수
# -------------------------------------
def check_whisper_cli():
    return os.path.exists(WHISPER_CLI)

def check_model_in_whisper():
    return os.path.exists(WHISPER_MODEL)

def check_coreml_encoder():
    # 미리 번들된 엔코더 파일이 복사되었는지 확인
    return os.path.exists(resource_path("whisper.cpp/models/ggml-large-v3-turbo-encoder.mlmodelc"))

def install_coreml_dependencies(progress_callback):
    progress_callback("Mac ARM: CoreML 의존성 설치 중...", 5)
    subprocess.run(["pip", "install", "ane_transformers", "openai-whisper", "coremltools"], check=True)
    try:
        subprocess.run(["xcode-select", "-p"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        progress_callback("Xcode 명령줄 도구 확인 완료", 10)
    except Exception:
        progress_callback("Xcode 명령줄 도구 미설치", 10)
        messagebox.showwarning("경고", "Xcode 명령줄 도구 미설치.\n터미널에서 'xcode-select --install'을 실행해주세요.")

# --- 미리 번들된 엔코더 및 모델 파일 복사 함수 ---
def copy_prebundled_files(progress_callback):
    """
    미리 번들된 엔코더 및 모델 파일을 resource 폴더에서
    whisper.cpp/models 폴더로 복사합니다.
    """
    models_dir = resource_path("whisper.cpp/models")
    os.makedirs(models_dir, exist_ok=True)
    encoder_src = resource_path("resources/ggml-large-v3-turbo-encoder.mlmodelc")
    model_src   = resource_path("resources/ggml-large-v3-turbo.bin")
    encoder_dest = os.path.join(models_dir, "ggml-large-v3-turbo-encoder.mlmodelc")
    model_dest   = os.path.join(models_dir, "ggml-large-v3-turbo.bin")
    
    if os.path.isdir(encoder_src):
        if not os.path.exists(encoder_dest):
            shutil.copytree(encoder_src, encoder_dest)
            progress_callback("미리 번들된 엔코더 디렉토리 복사 완료", 30)
        else:
            progress_callback("엔코더 디렉토리가 이미 존재합니다.", 30)
    else:
        if not os.path.exists(encoder_dest):
            shutil.copy2(encoder_src, encoder_dest)
            progress_callback("미리 번들된 엔코더 파일 복사 완료", 30)
        else:
            progress_callback("엔코더 파일이 이미 존재합니다.", 30)
            
    if os.path.isdir(model_src):
        if not os.path.exists(model_dest):
            shutil.copytree(model_src, model_dest)
            progress_callback("미리 번들된 모델 디렉토리 복사 완료", 30)
        else:
            progress_callback("모델 디렉토리가 이미 존재합니다.", 30)
    else:
        if not os.path.exists(model_dest):
            shutil.copy2(model_src, model_dest)
            progress_callback("미리 번들된 모델 파일 복사 완료", 30)
        else:
            progress_callback("모델 파일이 이미 존재합니다.", 30)

def download_and_build_whisper(progress_callback):
    target_dir = resource_path("whisper.cpp")
    if not os.path.exists(target_dir):
        progress_callback("Whisper.cpp 저장소 클론 중...", 10)
        try:
            # 번들 내 바이너리 git을 사용하여 클론할 때, 환경변수 PATH를 전달합니다.
            subprocess.run(
                [git_path, "clone", "https://github.com/ggml-org/whisper.cpp.git"],
                check=True, env=env
            )
        except subprocess.CalledProcessError as e:
            progress_callback(f"Git 클론 실패: {e}", 10)
            # 클론 실패 시 미리 번들된 파일 복사로 대체
            progress_callback("미리 번들된 파일을 사용합니다.", 10)
            copy_prebundled_files(progress_callback)
    else:
        progress_callback("Whisper.cpp 저장소가 이미 존재합니다.", 10)
        
    copy_prebundled_files(progress_callback)
    
    os_type = platform.system()
    arch = platform.machine().lower()
    if os_type == "Darwin" and "arm" in arch:
        install_coreml_dependencies(progress_callback)
        progress_callback("CMake 구성 중 (Core ML 지원)...", 40)
        subprocess.run(["cmake", "-B", "build", "-DWHISPER_COREML=1"],
                       cwd=resource_path("whisper.cpp"), check=True, env=env)
        progress_callback("Core ML 지원 빌드 진행 중...", 50)
        subprocess.run(["cmake", "--build", "build", "-j", "--config", "Release"],
                       cwd=resource_path("whisper.cpp"), check=True, env=env)
        progress_callback("Whisper.cpp Core ML 지원 빌드 완료", 70)
    else:
        progress_callback("CMake 구성 중...", 40)
        subprocess.run(["cmake", "-B", "build"],
                       cwd=resource_path("whisper.cpp"), check=True, env=env)
        progress_callback("프로젝트 빌드 중...", 50)
        if os_type == "Windows":
            build_command = ["cmake", "--build", "build", "--config", "Release"]
        else:
            build_command = ["cmake", "--build", "build"]
        subprocess.run(build_command, cwd=resource_path("whisper.cpp"), check=True, env=env)
        progress_callback("Whisper.cpp 빌드 완료", 70)

def installation_process(progress_callback):
    try:
        progress_callback("시스템 환경 확인 중...", 0)
        os_type = platform.system()
        arch = platform.machine()
        progress_callback(f"운영체제: {os_type}, 아키텍처: {arch}", 5)
        
        # 시스템 의존성(예: Homebrew, ffmpeg, git, cmake)을 점검 및 설치
        check_and_install_system_dependencies(progress_callback)

        if not check_whisper_cli():
            progress_callback("Whisper CLI가 없습니다. 다운로드 및 빌드 진행...", 5)
            download_and_build_whisper(progress_callback)
        else:
            progress_callback("Whisper CLI가 이미 존재합니다.", 70)
        progress_callback("전체 설치 완료", 100)
        return True
    except Exception as e:
        progress_callback(f"오류 발생: {e}", 100)
        raise

# -------------------------------------
# 오디오 필사 관련 함수들
# -------------------------------------
def check_ffmpeg():
    if os.system(f'"{ffmpeg_path}" -version') != 0:
        audio_log_message("❌ FFmpeg가 설치되어 있지 않습니다.")
        messagebox.showerror("오류", "FFmpeg 미설치.")
        os._exit(1)

def convert_to_mp3(file_path):
    if file_path.lower().endswith(".mp3"):
        audio_log_message(f"🎵 이미 MP3: {file_path}")
        return file_path
    output_mp3 = file_path.rsplit(".", 1)[0] + ".mp3"
    audio_log_message(f"🔄 WAV → MP3: {file_path} → {output_mp3}")
    command = f'"{ffmpeg_path}" -i "{file_path}" -c:a libmp3lame -b:a 128k "{output_mp3}" -y'
    os.system(command)
    if not os.path.exists(output_mp3):
        audio_log_message(f"❌ MP3 변환 실패: {output_mp3}")
        os._exit(1)
    return output_mp3

def split_audio(file_path):
    check_ffmpeg()
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = os.path.join("split_audio", base_name)
    mp3_folder = os.path.join(output_folder, "MP3")
    text_folder = os.path.join(output_folder, "TEXT")
    os.makedirs(mp3_folder, exist_ok=True)
    os.makedirs(text_folder, exist_ok=True)
    file_path = convert_to_mp3(file_path)
    audio_log_message("🔍 Silero VAD 모델 로딩 중...")
    model, utils = torch.hub.load('snakers4/silero-vad', model='silero_vad')
    get_speech_timestamps = utils[0]
    read_audio = utils[2]
    audio_log_message(f"🎵 오디오 로드: {file_path}")
    wav = read_audio(file_path, sampling_rate=16000)
    audio_log_message("🧠 음성 구간 감지 중...")
    speech_timestamps = get_speech_timestamps(
        wav, 
        model, 
        sampling_rate=16000,
        min_speech_duration_ms=vad_config.get("min_speech_duration_ms", 500),
        min_silence_duration_ms=vad_config.get("min_silence_duration_ms", 700),
        max_speech_duration_s=vad_config.get("max_speech_duration_s", 18),
        speech_pad_ms=vad_config.get("speech_pad_ms", 10),
        threshold=vad_config.get("threshold", 0.6)
    )
    total_segments = len(speech_timestamps)
    audio_log_message(f"✅ 감지 구간: {total_segments} 개")
    for idx, segment in enumerate(speech_timestamps):
        start_time = max(0, segment['start'] / 16000)
        end_time = segment['end'] / 16000
        duration = end_time - start_time
        if start_time >= end_time or duration < 0.5:
            audio_log_message(f"⚠️ 스킵됨: {idx+1}.mp3 (잘못된 구간)")
            continue
        output_mp3 = os.path.join(mp3_folder, f"{idx+1}.mp3")
        command = f'"{ffmpeg_path}" -i "{file_path}" -ss {start_time} -t {duration} -c copy "{output_mp3}" -y'
        audio_log_message(f"🔪 MP3 분할: {idx+1}/{total_segments}")
        os.system(command)
    audio_log_message("✅ 분할 완료!")
    return output_folder, mp3_folder, text_folder

def remove_newlines_from_text(text_file):
    with open(text_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    cleaned_text = " ".join(line.strip() for line in lines)
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    audio_log_message(f"✅ 개행 제거: {text_file}")

def transcribe_audio(mp3_folder, text_folder):
    if not os.path.exists(mp3_folder):
        audio_log_message(f"❌ 폴더 없음: {mp3_folder}")
        os._exit(1)
    audio_files = sorted([os.path.join(mp3_folder, f) for f in os.listdir(mp3_folder) if f.endswith(".mp3")])
    total_files = len(audio_files)
    audio_log_message(f"🎤 Whisper 필사: {total_files}개 파일")
    whisper_command = [
        WHISPER_CLI,
        "--model", WHISPER_MODEL,
        "--language", "ko",
        "--output-txt"
    ] + audio_files
    audio_log_message("🚀 Whisper CLI 실행 중...")
    proc = subprocess.Popen(whisper_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(proc.stdout.readline, ""):
        audio_log_message(line.strip())
    proc.wait()
    for file in audio_files:
        txt_file = f"{file}.txt"
        if os.path.exists(txt_file):
            remove_newlines_from_text(txt_file)
            new_location = os.path.join(text_folder, os.path.basename(txt_file))
            shutil.move(txt_file, new_location)
            audio_log_message(f"✅ {os.path.basename(file)} → {new_location}")
        else:
            audio_log_message(f"❌ TXT 파일 생성 실패: {txt_file}")
    audio_log_message(f"🎉 필사 완료! 결과: {text_folder}")

def audio_update_progress(value):
    audio_progress_bar['value'] = value
    audio_tab.update_idletasks()

# -------------------------------------
# Queue 기반 로그 처리 함수
# -------------------------------------
def process_install_log_queue():
    try:
        while True:
            msg = install_log_queue.get_nowait()
            install_log_box.config(state='normal')
            install_log_box.insert(tk.END, msg + "\n")
            install_log_box.see(tk.END)
            install_log_box.config(state='disabled')
    except queue.Empty:
        pass
    root.after(200, process_install_log_queue)

def process_audio_log_queue():
    try:
        while True:
            msg = audio_log_queue.get_nowait()
            audio_log_box.config(state='normal')
            audio_log_box.insert(tk.END, msg + "\n")
            audio_log_box.see(tk.END)
            audio_log_box.config(state='disabled')
    except queue.Empty:
        pass
    root.after(200, process_audio_log_queue)

# -------------------------------------
# GUI 및 탭 관련 함수
# -------------------------------------
def set_audio_buttons_state(state):
    choose_button.config(state=state)
    process_button.config(state=state)
    audio_file_entry.config(state=state)

def process_audio_function():
    try:
        set_audio_buttons_state("disabled")
        audio_update_progress(10)
        selected_file = audio_file_entry.get()
        if not selected_file:
            messagebox.showwarning("경고", "오디오 파일을 먼저 선택하세요.")
            set_audio_buttons_state("normal")
            return
        audio_log_message("=== 오디오 처리 시작 ===")
        selected_file = convert_to_mp3(selected_file)
        audio_update_progress(20)
        output_folder, mp3_folder, text_folder = split_audio(selected_file)
        audio_update_progress(60)
        transcribe_audio(mp3_folder, text_folder)
        audio_update_progress(100)
        audio_log_message("=== 오디오 처리 완료 ===")
        messagebox.showinfo("완료", "오디오 처리 및 필사가 완료되었습니다!")
    except Exception as e:
        audio_log_message(f"오류: {e}")
        messagebox.showerror("오류", f"오디오 처리 중 오류:\n{e}")
    finally:
        set_audio_buttons_state("normal")

def choose_audio_file():
    file_path = filedialog.askopenfilename(
        title="오디오 파일 선택",
        filetypes=[("Audio Files", "*.wav *.mp3"), ("All Files", "*.*")]
    )
    if file_path:
        audio_file_entry.delete(0, tk.END)
        audio_file_entry.insert(0, file_path)

def reset_settings():
    for key, default_value in DEFAULT_CONFIG.items():
        settings_entries[key].delete(0, tk.END)
        settings_entries[key].insert(0, str(default_value))
    global vad_config
    vad_config = DEFAULT_CONFIG.copy()
    settings_log_message("설정 초기화 완료")

def save_settings():
    global vad_config
    try:
        vad_config["min_speech_duration_ms"] = int(settings_entries["min_speech_duration_ms"].get())
        vad_config["min_silence_duration_ms"] = int(settings_entries["min_silence_duration_ms"].get())
        vad_config["max_speech_duration_s"] = float(settings_entries["max_speech_duration_s"].get())
        vad_config["speech_pad_ms"] = int(settings_entries["speech_pad_ms"].get())
        vad_config["threshold"] = float(settings_entries["threshold"].get())
        save_config()
        settings_log_message("설정 저장 완료")
    except Exception as e:
        settings_log_message(f"설정 저장 오류: {e}")
        messagebox.showerror("설정 오류", f"설정 저장 중 오류 발생:\n{e}")

def settings_log_message(msg):
    settings_log_box.config(state='normal')
    settings_log_box.insert(tk.END, msg + "\n")
    settings_log_box.see(tk.END)
    settings_log_box.config(state='disabled')

def load_settings_into_entries():
    for key, entry in settings_entries.items():
        entry.delete(0, tk.END)
        entry.insert(0, str(vad_config.get(key, DEFAULT_CONFIG[key])))

# -------------------------------------
# 설치 완료 후 탭 전환 및 자동 설치 함수
# -------------------------------------
def auto_install():
    try:
        installation_update_progress("자동 설치 시작...", 0)
        installation_process(installation_update_progress)
        installation_update_progress("설치 완료!", 100)
        root.after(0, lambda: messagebox.showinfo("설치 완료", "설치가 완료되었습니다!"))
        root.after(0, lambda: (
            notebook.add(audio_tab, text="오디오 필사"),
            notebook.add(settings_tab, text="설정"),
            notebook.forget(install_tab)
        ))
    except Exception as e:
        root.after(0, lambda: messagebox.showerror("설치 오류", f"자동 설치 중 오류 발생:\n{e}"))

def installation_update_progress(message, value):
    install_progress_bar['value'] = value
    install_status_label.config(text=message)
    install_tab.update_idletasks()
    install_log_message(message)
    root.after(200)

def install_log_message(msg):
    install_log_queue.put(msg)

def audio_log_message(msg):
    audio_log_queue.put(msg)

# -------------------------------------
# GUI 통합 (단일 Tk() 사용)
# -------------------------------------
root = tk.Tk()
root.title("Whisper 설치, 설정 및 오디오 필사 도구")
root.geometry("800x700")
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# ----- 설치 관리자 탭 (시작 시에만 보임) -----
install_tab = ttk.Frame(notebook, padding=10)
notebook.add(install_tab, text="Whisper 설치")
install_status_label = ttk.Label(install_tab, text="설치 준비 중...")
install_status_label.pack(pady=5)
install_progress_bar = ttk.Progressbar(install_tab, orient='horizontal', length=500, mode='determinate')
install_progress_bar.pack(pady=5)
install_log_box = scrolledtext.ScrolledText(install_tab, width=80, height=15, state='disabled',
                                              background="white", foreground="black")
install_log_box.pack(padx=5, pady=5)

# ----- 오디오 필사 탭 (초기에는 생성만 해두고 Notebook에 추가하지 않음) -----
audio_tab = ttk.Frame(notebook, padding=10)
audio_file_entry = ttk.Entry(audio_tab, width=80)
audio_file_entry.pack(pady=5)
choose_button = ttk.Button(audio_tab, text="파일 선택", command=choose_audio_file)
choose_button.pack(pady=5)
process_button = ttk.Button(audio_tab, text="필사 실행", command=lambda: threading.Thread(target=process_audio_function, daemon=True).start())
process_button.pack(pady=5)
audio_progress_bar = ttk.Progressbar(audio_tab, orient='horizontal', length=500, mode='determinate')
audio_progress_bar.pack(pady=5)
audio_log_box = scrolledtext.ScrolledText(audio_tab, width=80, height=15, state='disabled',
                                            background="white", foreground="black")
audio_log_box.pack(padx=5, pady=5, fill="both", expand=True)

# ----- 설정 탭 (초기에는 생성만 해두고 Notebook에 추가하지 않음) -----
settings_tab = ttk.Frame(notebook, padding=10)
settings_info = {
    "min_speech_duration_ms": "최소 음성 지속시간 (밀리초): (기본: 500)",
    "min_silence_duration_ms": "최소 침묵 지속시간 (밀리초): (기본: 700)",
    "max_speech_duration_s": "최대 음성 지속시간 (초): (기본: 18)",
    "speech_pad_ms": "음성 패딩 (밀리초): (기본: 10)",
    "threshold": "음성 검출 임계치: (기본: 0.6)"
}
settings_entries = {}
for key, info in settings_info.items():
    frame = ttk.Frame(settings_tab)
    frame.pack(fill="x", pady=2)
    label = ttk.Label(frame, text=info)
    label.pack(side="top", anchor="w")
    entry = ttk.Entry(frame)
    entry.pack(side="top", fill="x", expand=True)
    settings_entries[key] = entry
settings_button_frame = ttk.Frame(settings_tab)
settings_button_frame.pack(pady=5, fill="x")
reset_button = ttk.Button(settings_button_frame, text="초기화", command=reset_settings)
reset_button.pack(side="left", padx=5)
save_button = ttk.Button(settings_button_frame, text="저장", command=save_settings)
save_button.pack(side="left", padx=5)
settings_log_box = scrolledtext.ScrolledText(settings_tab, width=80, height=8, state='disabled',
                                               background="white", foreground="black")
settings_log_box.pack(padx=5, pady=5, fill="both", expand=True)
load_settings_into_entries()

# ----- 로그 큐 처리 시작 -----
root.after(200, process_install_log_queue)
root.after(200, process_audio_log_queue)

# ----- 자동 설치 시작 (백그라운드 스레드로 실행) -----
threading.Thread(target=auto_install, daemon=True).start()

root.mainloop()