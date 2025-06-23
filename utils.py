import subprocess
import queue

# 로그 큐들
install_log_queue = queue.Queue()
audio_log_queue = queue.Queue()


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
            stderr=subprocess.STDOUT,
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


def clear_mps_cache():
    """호환성을 위한 빈 함수 (MPS 사용 안함)"""
    import gc
    gc.collect()


def is_video_file(file_path):
    """파일이 영상 파일인지 확인"""
    import os
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    return os.path.splitext(file_path.lower())[1] in video_extensions


def is_audio_file(file_path):
    """파일이 음성 파일인지 확인"""
    import os
    audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
    return os.path.splitext(file_path.lower())[1] in audio_extensions
