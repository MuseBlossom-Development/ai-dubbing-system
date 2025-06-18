import os

# PyTorch 디바이스 설정 (CPU/CUDA만, MPS 제외)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torchaudio
import torch
import random
import numpy as np
import sys
import logging
import librosa


# 프로젝트 내 CosyVoice2 모델 로컬 경로 설정
repo_root = os.path.dirname(__file__)
LOCAL_COSYVOICE_MODEL = os.path.join(
    repo_root, 'CosyVoice', 'pretrained_models', 'CosyVoice2-0.5B'
)

# CosyVoice2 패키지 경로 추가
sys.path.insert(0, os.path.join(repo_root, 'CosyVoice'))
from cosyvoice.cli.cosyvoice import CosyVoice2

# 로깅 설정 (Gradio 앱과 동일)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Device-aware memory cleanup utility
def cleanup_memory(device):
    """Clean up memory for the current device"""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logging.debug("CUDA cache cleared")
    # CPU doesn't need explicit cleanup


# Gradio 앱의 postprocess 함수
def postprocess(speech: torch.Tensor,
                top_db: int = 60,
                hop_length: int = 220,
                win_length: int = 440,
                max_val: float = 0.8) -> torch.Tensor:
    """WebUI와 동일한 방식으로 prompt 오디오만 전처리"""
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(16000 * 0.2))], dim=1)  # 16kHz 기준 패딩
    return speech

# 오디오 로드 및 리샘플 함수
def load_wav_resample(path: str, target_sr: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=target_sr
        )
    # 스테레오→모노
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


# 오디오 분위기 분석 함수 추가
def analyze_audio_mood(audio_path: str) -> str:
    """
    오디오 파일을 분석해서 적절한 instruct 명령어를 반환합니다.
    """
    try:
        # 오디오 로드 (librosa 사용)
        y, sr = librosa.load(audio_path, sr=16000)

        # 1. 음성 특성 분석
        # 음성 강도 (RMS)
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = np.mean(rms)

        # 스펙트럴 중심 (음색 분석)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_spectral_centroid = np.mean(spectral_centroids)

        # 영교차율 (음성의 거칠기)
        zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
        avg_zcr = np.mean(zero_crossings)

        # MFCC 계수로 음성 특성 분석
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.var(mfccs, axis=1).mean()  # MFCC 변화량

        # 2. 분위기 판단 로직 (템포 분석 제거)
        if avg_rms > 0.05 and avg_spectral_centroid > 2000:
            return "활기차게 말해"
        elif avg_rms < 0.02 and avg_spectral_centroid < 1500:
            return "차분하게 말해"
        elif avg_zcr > 0.1 or avg_rms > 0.08:
            return "감정적으로 말해"
        elif mfcc_var < 50:
            return "천천히 말해"
        elif mfcc_var > 150:
            return "빠르게 말해"
        else:
            return "자연스럽게 말해"

    except Exception as e:
        logging.warning(f"오디오 분위기 분석 실패: {e}")
        return "자연스럽게 말해"


# 배치 합성 함수
def main(audio_dir, prompt_text_dir, text_dir, out_dir, model_path=LOCAL_COSYVOICE_MODEL, enable_instruct=True,
         manual_command=None):
    # Device 설정 (MPS 지원 제거)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA GPU acceleration")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # 모델 초기화
    global cosy
    cosy = CosyVoice2(
        model_path,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=False
    )

    # 입력 파일 목록
    audio_files  = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')])
    prompt_files = sorted([f for f in os.listdir(prompt_text_dir) if f.lower().endswith('.txt')])
    text_files   = sorted([f for f in os.listdir(text_dir) if f.lower().endswith('.txt')])

    logging.info(f"[DEBUG] Audio files      ({len(audio_files)}): {audio_files}")
    logging.info(f"[DEBUG] Prompt Text files ({len(prompt_files)}): {prompt_files}")
    logging.info(f"[DEBUG] Text files       ({len(text_files)}): {text_files}")

    if not (len(audio_files) == len(prompt_files) == len(text_files)):
        raise ValueError("오디오, 프롬프트 텍스트, 목표 텍스트 파일 수가 일치하지 않습니다.")

    # 출력 디렉토리 생성
    zero_shot_dir = os.path.join(out_dir, 'zero_shot')
    instruct_dir = os.path.join(out_dir, 'instruct')
    os.makedirs(zero_shot_dir, exist_ok=True)
    os.makedirs(instruct_dir, exist_ok=True)

    current_seed = random.randint(0, 2**32 - 1)
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)
    logging.info(f"Using random seed: {current_seed} on device: {device}")

    # 파일별 합성
    for i, (awav, ptxt, txt) in enumerate(zip(audio_files, prompt_files, text_files), 1):
        wav_path   = os.path.join(audio_dir, awav)
        ptxt_path  = os.path.join(prompt_text_dir, ptxt)
        txt_path   = os.path.join(text_dir, txt)

        # 로깅
        logging.info(f"[{i}] Processing → {awav} / {txt}")

        # 오디오 & 텍스트 로드
        prompt_wav = load_wav_resample(wav_path)

        with open(ptxt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # 원본 오디오 분위기 분석 (Instruct2 활성화 시에만)
        if enable_instruct:
            if manual_command:
                instruct_command = manual_command
            else:
                instruct_command = analyze_audio_mood(wav_path)
            logging.info(f"  → 분석된 음성 스타일: '{instruct_command}'")
        else:
            instruct_command = None

        # 파일명에서 기본 이름과 세그먼트 번호 추출
        base_name = os.path.splitext(awav)[0]  # 예: "조용석_1m_001"
        if '_' in base_name:
            # "조용석_1m_001"에서 "조용석_1m"과 "001" 분리
            parts = base_name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                audio_base = parts[0]  # "조용석_1m"
                segment_num = parts[1]  # "001"
            else:
                audio_base = base_name
                segment_num = f"{i:03d}"
        else:
            audio_base = base_name
            segment_num = f"{i:03d}"

        try:
            # Prompt 오디오 전처리 (WebUI와 동일)
            prompt_wav_processed = postprocess(prompt_wav)

            # 1. Zero-shot 합성
            logging.info(f"  → Zero-shot 합성 중...")
            results_zero = cosy.inference_zero_shot(
                text,
                prompt_text,
                prompt_wav_processed,  # 전처리된 prompt 사용
                "",
                stream=False,
                text_frontend=True,
                speed=1
            )

            # Zero-shot 결과 저장 (병합 함수가 기대하는 파일명으로)
            for idx, out in enumerate(results_zero):
                speech = out['tts_speech']
                # 텐서를 CPU로 이동 후 저장
                if speech.device.type != 'cpu':
                    speech = speech.cpu()
                fname = f"{audio_base}_{segment_num}.wav"  # 예: "조용석_1m_001.wav"
                save_path = os.path.join(out_dir, fname)  # zero_shot_dir 대신 out_dir 사용
                torchaudio.save(save_path, speech, 24000)
                logging.info(f"    Zero-shot saved ➜ {fname}")

            # 2. Instruct2 합성은 선택사항으로 별도 폴더에 저장
            if enable_instruct:
                logging.info(f"  → Instruct2 합성 중: '{instruct_command}'")
                try:
                    results_instruct = cosy.inference_instruct2(
                        text,
                        instruct_command,
                        prompt_wav_processed,  # 전처리된 prompt 사용
                        stream=False
                    )

                    # Instruct2 결과 저장 (별도 폴더)
                    for idx, out in enumerate(results_instruct):
                        speech = out['tts_speech']
                        # 텐서를 CPU로 이동 후 저장
                        if speech.device.type != 'cpu':
                            speech = speech.cpu()
                        fname = f"{audio_base}_{segment_num}.wav"  # 같은 파일명으로 저장
                        save_path = os.path.join(instruct_dir, fname)
                        torchaudio.save(save_path, speech, 24000)
                        logging.info(f"    Instruct2 saved ➜ {fname}")

                except Exception as e:
                    logging.error(f"    Instruct2 실패 ({instruct_command}): {e}")

        except Exception as e:
            logging.error(f"Failed processing {awav}/{txt}: {e}")

        # 메모리 정리 (각 파일 처리 후)
        cleanup_memory(device)

    logging.info(f"✅ 배치 처리 완료 - 총 {len(audio_files)}개 파일 처리됨")

    # 최종 메모리 정리
    cleanup_memory(device)


# 스크립트 엔트리포인트
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Batch zero-shot TTS with CosyVoice2 (Gradio-matching)"
    )
    parser.add_argument('--audio_dir',       required=True, help="참조 .wav 폴더 경로")
    parser.add_argument('--prompt_text_dir', required=True, help="프롬프트 텍스트(.txt) 폴더 경로")
    parser.add_argument('--text_dir',        required=True, help="합성 텍스트(.txt) 폴더 경로")
    parser.add_argument('--out_dir',         required=True, help="결과 WAV 저장 폴더")
    parser.add_argument('--model_path',      default=LOCAL_COSYVOICE_MODEL, help="CosyVoice2 모델 경로")
    parser.add_argument('--enable_instruct', action='store_true', default=False, help="Instruct2 기능 활성화")
    parser.add_argument('--manual_command', type=str, default=None, help="수동으로 지정할 instruct 명령어")
    args = parser.parse_args()

    main(
        args.audio_dir,
        args.prompt_text_dir,
        args.text_dir,
        args.out_dir,
        model_path=args.model_path,
        enable_instruct=args.enable_instruct,
        manual_command=args.manual_command
    )
