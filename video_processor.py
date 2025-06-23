import os
import sys
import shutil
import subprocess
from math import log10
from pydub import AudioSegment
from config import get_ffmpeg_path
from utils import log_message, run_command_with_logging

# 현재 디렉터리를 기준으로 repo_root 설정
repo_root = os.path.dirname(__file__)


def log_message(message, also_print=True):
    """로그 메시지 출력 - 메인 모듈에서 재정의됨"""
    print(f"[VIDEO] {message}")


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

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                if line:
                    log_message(line)

        return_code = process.poll()
        log_message(f"{description} 완료 (return code: {return_code})")
        return return_code

    except Exception as e:
        log_message(f"{description} 오류: {e}")
        return -1


def extract_audio_from_video(video_path, output_audio_path):
    """FFmpeg를 사용해 영상에서 음성을 추출"""
    try:
        ffmpeg_path = get_ffmpeg_path()
        cmd = [
            ffmpeg_path, '-i', video_path,
            '-vn',  # 비디오 스트림 제외
            '-acodec', 'pcm_s16le',  # WAV 형식으로 추출
            '-ar', '44100',  # 샘플레이트 44.1kHz
            '-ac', '2',  # 스테레오
            '-y',  # 덮어쓰기 허용
            output_audio_path
        ]

        return_code = run_command_with_logging(cmd, description="영상에서 음성 추출")
        if return_code == 0:
            log_message(f"✅ 음성 추출 완료: {output_audio_path}")
            return True
        else:
            log_message(f"❌ 음성 추출 실패: return code {return_code}")
            return False

    except Exception as e:
        log_message(f"음성 추출 오류: {e}")
        return False


def separate_vocals_background(audio_path, output_dir):
    """UVR5를 사용해 음성과 배경음을 분리"""
    try:
        # UVR5 경로 설정
        uvr5_path = os.path.join(repo_root, 'GPT-SoVITS', 'tools', 'uvr5')
        if not os.path.exists(uvr5_path):
            log_message(f"❌ UVR5 경로를 찾을 수 없습니다: {uvr5_path}")
            return None, None

        log_message(f"🎵 UVR5를 사용해 음성 분리 시작: {audio_path}")

        # UVR5 모듈 경로를 sys.path에 추가
        if uvr5_path not in sys.path:
            sys.path.insert(0, uvr5_path)

        # GPT-SoVITS tools 경로도 추가
        tools_path = os.path.join(repo_root, 'GPT-SoVITS', 'tools')
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)

        try:
            # UVR5 모듈들 import
            from bsroformer import Roformer_Loader
            from mdxnet import MDXNetDereverb
            from vr import AudioPre, AudioPreDeEcho
            import torch

            # 디바이스 설정
            device = torch.device('cpu')

            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.xpu.is_available():
                device = torch.device('xpu')

            is_half = False  # 안정성을 위해 False로 설정

            # 모델명 설정 (기본적으로 HP2 사용 - 인성 보존에 좋음)
            model_name = "HP2_all_vocals"
            weight_uvr5_root = os.path.join(uvr5_path, "uvr5_weights")

            # 출력 디렉토리 설정
            vocals_dir = os.path.join(output_dir, "vocals")
            background_dir = os.path.join(output_dir, "background")
            os.makedirs(vocals_dir, exist_ok=True)
            os.makedirs(background_dir, exist_ok=True)

            # 파일명 설정
            base_name = os.path.splitext(os.path.basename(audio_path))[0]

            log_message(f"🔧 UVR5 설정: 모델={model_name}, 디바이스={device}")

            # AudioPre 객체 생성
            pre_fun = AudioPre(
                agg=10,  # 인성 추출 강도
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )

            # 음성 분리 실행
            format0 = "wav"
            is_hp3 = "HP3" in model_name

            log_message("🎵 음성 분리 처리 중...")
            pre_fun._path_audio_(audio_path, background_dir, vocals_dir, format0, is_hp3)

            # 결과 파일 경로 찾기
            vocals_path = None
            background_path = None

            # vocals 폴더에서 파일 찾기
            if os.path.exists(vocals_dir):
                for f in os.listdir(vocals_dir):
                    if f.endswith('.wav') and base_name in f:
                        vocals_path = os.path.join(vocals_dir, f)
                        break

            # background 폴더에서 파일 찾기  
            if os.path.exists(background_dir):
                for f in os.listdir(background_dir):
                    if f.endswith('.wav') and base_name in f:
                        background_path = os.path.join(background_dir, f)
                        break

            # 파일명이 예상과 다를 수 있으므로 대안적 방법으로 찾기
            if not vocals_path or not background_path:
                # vocals 폴더의 모든 wav 파일 중 가장 최근 파일
                if os.path.exists(vocals_dir):
                    wav_files = [f for f in os.listdir(vocals_dir) if f.endswith('.wav')]
                    if wav_files:
                        vocals_path = os.path.join(vocals_dir, wav_files[0])

                # background 폴더의 모든 wav 파일 중 가장 최근 파일
                if os.path.exists(background_dir):
                    wav_files = [f for f in os.listdir(background_dir) if f.endswith('.wav')]
                    if wav_files:
                        background_path = os.path.join(background_dir, wav_files[0])

            # 정리
            del pre_fun.model
            del pre_fun
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if vocals_path and background_path and os.path.exists(vocals_path) and os.path.exists(background_path):
                log_message(f"✅ 음성 분리 완료:")
                log_message(f"   🎤 보컬: {vocals_path}")
                log_message(f"   🎵 배경음: {background_path}")
                return vocals_path, background_path
            else:
                log_message("❌ 음성 분리 결과 파일을 찾을 수 없습니다")
                log_message(
                    f"   vocals_path: {vocals_path} (존재: {os.path.exists(vocals_path) if vocals_path else False})")
                log_message(
                    f"   background_path: {background_path} (존재: {os.path.exists(background_path) if background_path else False})")
                return None, None

        except ImportError as e:
            log_message(f"❌ UVR5 모듈 import 실패: {e}")
            log_message("필요한 의존성이 설치되지 않았을 수 있습니다.")
            return None, None

    except Exception as e:
        log_message(f"음성 분리 오류: {e}")
        import traceback
        log_message(f"상세 오류: {traceback.format_exc()}")
        return None, None


def combine_processed_audio_with_background(vocals_path, background_path, output_path, vocals_volume=1.0,
                                            background_volume=0.8):
    """처리된 보컬과 원본 배경음을 합성"""
    try:
        log_message(f"🎵 음성 합성 시작:")
        log_message(f"   🎤 보컬: {vocals_path}")
        log_message(f"   🎵 배경음: {background_path}")

        # 오디오 로드
        vocals = AudioSegment.from_file(vocals_path)
        background = AudioSegment.from_file(background_path)

        # 볼륨 조절
        if vocals_volume != 1.0:
            vocals = vocals + (20 * log10(vocals_volume))  # dB 변환
        if background_volume != 1.0:
            background = background + (20 * log10(background_volume))

        # 길이 맞춤 (더 긴 쪽에 맞춤)
        max_length = max(len(vocals), len(background))

        if len(vocals) < max_length:
            # 보컬이 짧으면 끝에 무음 추가
            vocals = vocals + AudioSegment.silent(duration=max_length - len(vocals))
        elif len(background) < max_length:
            # 배경음이 짧으면 루프 또는 무음 추가
            if len(background) > 0:
                # 배경음을 루프해서 길이 맞춤
                loops_needed = (max_length // len(background)) + 1
                background = background * loops_needed
                background = background[:max_length]
            else:
                background = AudioSegment.silent(duration=max_length)

        # 오디오 합성
        combined = vocals.overlay(background)

        # 결과 저장
        combined.export(output_path, format="wav")
        log_message(f"✅ 음성 합성 완료: {output_path}")
        return True

    except Exception as e:
        log_message(f"음성 합성 오류: {e}")
        return False


def combine_audio_with_video(video_path, audio_path, output_video_path):
    """음성과 영상을 합쳐서 최종 영상 생성"""
    try:
        log_message(f"🎬 영상 합성 시작:")
        log_message(f"   📹 영상: {video_path}")
        log_message(f"   🎵 음성: {audio_path}")

        ffmpeg_path = get_ffmpeg_path()
        cmd = [
            ffmpeg_path,
            '-i', video_path,  # 입력 영상
            '-i', audio_path,  # 입력 음성
            '-c:v', 'copy',  # 비디오 코덱 복사 (re-encoding 안함)
            '-c:a', 'aac',  # 오디오 코덱
            '-b:a', '192k',  # 오디오 비트레이트
            '-map', '0:v:0',  # 첫 번째 입력의 비디오 스트림
            '-map', '1:a:0',  # 두 번째 입력의 오디오 스트림
            '-shortest',  # 더 짧은 스트림에 맞춤
            '-y',  # 덮어쓰기 허용
            output_video_path
        ]

        return_code = run_command_with_logging(cmd, description="영상과 음성 합성")
        if return_code == 0:
            log_message(f"✅ 영상 합성 완료: {output_video_path}")
            return True
        else:
            log_message(f"❌ 영상 합성 실패: return code {return_code}")
            return False

    except Exception as e:
        log_message(f"영상 합성 오류: {e}")
        return False


def is_video_file(file_path):
    """파일이 영상 파일인지 확인"""
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    return os.path.splitext(file_path.lower())[1] in video_extensions


def is_audio_file(file_path):
    """파일이 음성 파일인지 확인"""
    audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
    return os.path.splitext(file_path.lower())[1] in audio_extensions


def process_video_file(input_video_path, output_dir):
    """
    영상 파일을 처리하여 음성 추출 및 보컬 분리 수행
    
    Returns:
        tuple: (extracted_audio_path, vocals_path, background_path, original_video_path)
    """
    try:
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]

        # 음성 추출
        extracted_audio_path = os.path.join(output_dir, f"{base_name}_extracted.wav")
        if not extract_audio_from_video(input_video_path, extracted_audio_path):
            return None, None, None, input_video_path

        # 보컬/배경음 분리
        separation_dir = os.path.join(output_dir, "separation")
        os.makedirs(separation_dir, exist_ok=True)

        vocals_path, background_path = separate_vocals_background(extracted_audio_path, separation_dir)

        if vocals_path and background_path:
            log_message(f"✅ 영상 처리 완료:")
            log_message(f"   📹 원본 영상: {input_video_path}")
            log_message(f"   🎵 추출된 음성: {extracted_audio_path}")
            log_message(f"   🎤 분리된 보컬: {vocals_path}")
            log_message(f"   🎵 분리된 배경음: {background_path}")
            return extracted_audio_path, vocals_path, background_path, input_video_path
        else:
            log_message("❌ 보컬/배경음 분리 실패")
            return extracted_audio_path, None, None, input_video_path

    except Exception as e:
        log_message(f"영상 처리 오류: {e}")
        return None, None, None, input_video_path
