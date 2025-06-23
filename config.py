import os
import json
import platform

# OS/환경 정보
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"

# 설정 파일 경로
CONFIG_FILE = 'vad_config.json'

# VAD 기본 설정
DEFAULT_VAD_CONFIG = {
    'threshold': 0.5,  # 음성 감지 민감도
    'min_speech_duration_ms': 3500,  # 최소 음성 구간 1초
    'max_speech_duration_s': 15.0,  # 최대 음성 구간 30초
    'min_silence_duration_ms': 500,  # 최소 무음 구간 0.5초
    'speech_pad_ms': 10,  # 음성 패딩 100ms
}


def resource_path(relative_path):
    """리소스 파일 경로 반환"""
    try:
        import sys
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.getcwd()
    if IS_WINDOWS and relative_path.endswith('whisper-cli'):
        relative_path += '.exe'
    return os.path.join(base_path, relative_path)


def load_vad_config():
    """VAD 설정 로드"""
    path = resource_path(CONFIG_FILE)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
    else:
        loaded = {}

    config = DEFAULT_VAD_CONFIG.copy()
    config.update(loaded)
    return config


def save_vad_config(config):
    """VAD 설정 저장"""
    try:
        with open(resource_path(CONFIG_FILE), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"설정 저장 오류: {e}")
        return False


def get_whisper_cli_path():
    """Whisper CLI 경로 반환"""
    if IS_WINDOWS:
        whisper_cli = resource_path('whisper.cpp/build/bin/Release/whisper-cli')
        if not os.path.exists(whisper_cli):
            whisper_cli = resource_path('whisper.cpp/build/bin/whisper-cli')
    else:
        whisper_cli = resource_path('whisper.cpp/build/bin/whisper-cli')
    return whisper_cli


def get_model_path():
    """OS에 따른 적절한 모델 경로 반환"""
    if IS_MACOS:
        # macOS: 먼저 CoreML 모델 확인, 없으면 GGML 모델 사용
        coreml_path = resource_path('whisper.cpp/models/ggml-large-v3-turbo-encoder.mlmodelc')
        ggml_path = resource_path('whisper.cpp/models/ggml-large-v3-turbo.bin')

        if os.path.exists(coreml_path) and os.path.exists(ggml_path):
            return ggml_path, True  # GGML 모델 경로, CoreML 사용
        elif os.path.exists(ggml_path):
            return ggml_path, False
        else:
            return resource_path('whisper.cpp/models/for-tests-ggml-base.bin'), False
    else:
        # Windows/Linux: GGML 모델만 사용
        ggml_path = resource_path('resources/ggml-large-v3-turbo.bin')
        if os.path.exists(ggml_path):
            return ggml_path, False
        else:
            return resource_path('whisper.cpp/models/ggml-large-v3-turbo.bin'), False


def get_ffmpeg_path():
    """FFmpeg 경로 탐색"""
    import shutil
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path and IS_WINDOWS:
        candidate = os.path.join(os.environ.get("CONDA_PREFIX", ""), "Library", "bin", "ffmpeg.exe")
        if os.path.exists(candidate):
            ffmpeg_path = candidate
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg 실행파일을 찾을 수 없습니다.")
    return ffmpeg_path
