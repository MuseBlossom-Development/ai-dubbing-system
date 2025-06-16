import os
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

# Gradio 앱의 postprocess 함수
def postprocess(speech: torch.Tensor,
                top_db: int = 60,
                hop_length: int = 220,
                win_length: int = 440,
                max_val: float = 0.8,
                pad_sec: float = 0.2) -> torch.Tensor:
    # 앞뒤 무음 트리밍
    arr = speech.squeeze(0).cpu().numpy()
    trimmed, _ = librosa.effects.trim(
        arr, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    # 정규화
    trimmed = trimmed / np.maximum(np.abs(trimmed).max(), 1e-9) * max_val
    # 무음 패딩
    pad = np.zeros(int(pad_sec * cosy.sample_rate), dtype=np.float32)
    out = np.concatenate([trimmed, pad], axis=0)
    return torch.from_numpy(out).unsqueeze(0)

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

# 배치 합성 함수
def main(audio_dir, prompt_text_dir, text_dir, out_dir, model_path=LOCAL_COSYVOICE_MODEL):
    # 모델 초기화
    global cosy
    cosy = CosyVoice2(
        model_path,
        load_jit=False,
        load_trt=False,
        fp16=False,
        use_flow_cache=False
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

    os.makedirs(out_dir, exist_ok=True)


    current_seed = random.randint(0, 2**32 - 1)
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)
    logging.info(f"Using random seed: {current_seed}")


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

        try:
            # 합성 실행 (stream=False or True 둘 다 가능)
            results = cosy.inference_zero_shot(
                text,
                prompt_text,
                prompt_wav,
                "",           # speaker_id
                stream=False,  # <-- 수정된 부분
                text_frontend=True,
                speed=1.05
            )

            # 결과 저장
            for idx, out in enumerate(results):
                speech = out['tts_speech']
                # speech = postprocess(speech)
                fname = f"{i:03d}.wav"
                save_path = os.path.join(out_dir, fname)
                torchaudio.save(save_path, speech, 24000)
                logging.info(f"[{i}/{len(audio_files)}] saved ➜ {fname}")

        except Exception as e:
            logging.error(f"Failed processing {awav}/{txt}: {e}")

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
    args = parser.parse_args()

    main(
        args.audio_dir,
        args.prompt_text_dir,
        args.text_dir,
        args.out_dir,
        model_path=args.model_path
    )
