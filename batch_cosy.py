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

# 파일명 안전화 함수 임포트
from audio_processor import sanitize_filename, safe_file_operations

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


def cleanup_cosyvoice_model():
    """CosyVoice 모델 메모리 해제"""
    global cosy
    if 'cosy' in globals() and cosy is not None:
        try:
            # 모델 구성 요소들 메모리 해제
            if hasattr(cosy, 'model') and cosy.model is not None:
                # LLM 모델 해제
                if hasattr(cosy.model, 'llm') and cosy.model.llm is not None:
                    del cosy.model.llm

                # Flow 모델 해제
                if hasattr(cosy.model, 'flow') and cosy.model.flow is not None:
                    del cosy.model.flow

                # Hift 모델 해제
                if hasattr(cosy.model, 'hift') and cosy.model.hift is not None:
                    del cosy.model.hift

                del cosy.model

            # Frontend 해제
            if hasattr(cosy, 'frontend') and cosy.frontend is not None:
                # ONNX 세션 해제
                if hasattr(cosy.frontend, 'campplus_session') and cosy.frontend.campplus_session is not None:
                    del cosy.frontend.campplus_session

                if hasattr(cosy.frontend,
                           'speech_tokenizer_session') and cosy.frontend.speech_tokenizer_session is not None:
                    del cosy.frontend.speech_tokenizer_session

                del cosy.frontend

            # CosyVoice 객체 해제
            del cosy
            cosy = None

            logging.info("✅ CosyVoice 모델 메모리 해제 완료")

        except Exception as e:
            logging.error(f"⚠️ CosyVoice 모델 해제 중 오류: {e}")

        # 가비지 컬렉션 및 CUDA 캐시 정리
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logging.info("🔧 CUDA 메모리 캐시 정리 완료")


def synthesize_with_speed_control(cosy, text, prompt_text, prompt_wav, target_speed=1.0,
                                  target_language='korean', instruct_command=None):
    """
    CosyVoice2의 네이티브 speed 파라미터를 활용한 음성 합성

    Args:
        cosy: CosyVoice2 모델
        text: 합성할 텍스트
        prompt_text: 프롬프트 텍스트
        prompt_wav: 프롬프트 오디오
        target_speed: 목표 속도 (0.5 = 매우느림, 1.0 = 보통, 1.5 = 빠름, 2.0 = 매우빠름)
        target_language: 타겟 언어
        instruct_command: 사용자 정의 instruct 명령어 (보조적 사용)

    Returns:
        합성된 오디오 텐서
    """
    logging.info(f"  🎵 네이티브 속도 제어 합성: {target_speed:.1f}배 속도")

    # 1단계: Zero-shot with native speed control (주 방식)
    try:
        results = cosy.inference_zero_shot(
            text,
            prompt_text,
            prompt_wav,
            "",
            stream=True,
            speed=target_speed,  # CosyVoice2의 네이티브 speed 파라미터 활용
            text_frontend=True
        )

        if results is None:
            logging.error("  ❌ Zero-shot 속도 제어 합성 실패")
            return None

        # 결과 연결
        result_list = list(results)
        if not result_list:
            logging.error("  ❌ Zero-shot 속도 제어 합성 결과 없음")
            return None

        combined_audio = []
        for out in result_list:
            if 'tts_speech' in out:
                speech = out['tts_speech']
                if speech.device.type != 'cpu':
                    speech = speech.cpu()
                combined_audio.append(speech)

        if not combined_audio:
            logging.error("  ❌ 유효한 Zero-shot 속도 제어 합성 결과 없음")
            return None

        final_audio = torch.cat(combined_audio, dim=1)
        duration = final_audio.size(1) / 24000
        logging.info(f"  ✅ Zero-shot 네이티브 속도 제어 완료: {duration:.2f}초")

        return final_audio

    except Exception as e:
        logging.warning(f"  ⚠️ Zero-shot 네이티브 속도 제어 실패, Instruct2로 대체: {e}")

    # 2단계: Instruct2 with speed control (대체 방식)
    if instruct_command is None:
        if target_speed <= 0.7:
            base_command = '매우 천천히 말해'
        elif target_speed <= 0.9:
            base_command = '천천히 말해'
        elif target_speed >= 1.3:
            base_command = '빠르게 말해'
        elif target_speed >= 1.6:
            base_command = '매우 빠르게 말해'
        else:
            base_command = '자연스럽게 말해'

        instruct_command = get_language_specific_instruct_command(base_command, target_language)

    logging.info(f"  💬 대체 Instruct 명령어: '{instruct_command}'")

    try:
        results = cosy.inference_instruct2(
            text,
            instruct_command,
            prompt_wav,
            stream=True,
            speed=target_speed  # Instruct2에서도 네이티브 speed 파라미터 함께 사용
        )

        if results is None:
            logging.error("  ❌ Instruct2 속도 제어 합성 실패")
            return None

        # 결과 연결
        result_list = list(results)
        if not result_list:
            logging.error("  ❌ Instruct2 속도 제어 합성 결과 없음")
            return None

        combined_audio = []
        for out in result_list:
            if 'tts_speech' in out:
                speech = out['tts_speech']
                if speech.device.type != 'cpu':
                    speech = speech.cpu()
                combined_audio.append(speech)

        if not combined_audio:
            logging.error("  ❌ 유효한 Instruct2 속도 제어 합성 결과 없음")
            return None

        final_audio = torch.cat(combined_audio, dim=1)
        duration = final_audio.size(1) / 24000
        logging.info(f"  ✅ Instruct2 속도 제어 완료: {duration:.2f}초")

        return final_audio

    except Exception as e:
        logging.error(f"  ❌ Instruct2 속도 제어 합성 중 오류: {e}")
        return None


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
def load_wav_resample(path: str, target_sr: int = 16000, min_duration: float = 3.0) -> torch.Tensor:
    """
    오디오 로드 및 리샘플링 (CosyVoice 3초 제약 우회)

    Args:
        path: 오디오 파일 경로
        target_sr: 목표 샘플링 레이트
        min_duration: 최소 길이 (초) - CosyVoice 제약 우회용
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=target_sr
        )
    # 스테레오→모노
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 3초 제약 우회: 짧은 세그먼트에 무음 패딩 추가
    current_duration = waveform.size(1) / target_sr
    if current_duration < min_duration:
        needed_samples = int((min_duration - current_duration) * target_sr)
        # 자연스러운 무음 패딩 (끝에 추가)
        padding = torch.zeros(1, needed_samples)
        waveform = torch.cat([waveform, padding], dim=1)
        logging.debug(f"Padding added: {current_duration:.2f}s → {min_duration:.2f}s")

    return waveform


def optimize_prompt_audio(prompt_wav: torch.Tensor, target_sr: int = 16000) -> torch.Tensor:
    """
    프롬프트 오디오 최적화 - 늘어짐 방지

    Args:
        prompt_wav: 프롬프트 오디오 텐서
        target_sr: 샘플링 레이트

    Returns:
        최적화된 프롬프트 오디오
    """
    # 음성 활동 구간만 추출하여 집중도 향상
    if prompt_wav.size(1) > target_sr * 10:  # 10초 초과시 트림
        # 중간 부분 선택 (시작/끝 1초씩 제외)
        start_sample = target_sr  # 1초
        end_sample = min(prompt_wav.size(1) - target_sr, start_sample + target_sr * 8)  # 최대 8초
        prompt_wav = prompt_wav[:, start_sample:end_sample]
        logging.debug(f"Prompt trimmed: {prompt_wav.size(1) / target_sr:.2f}s")

    return prompt_wav


def smart_synthesis_with_length_control(cosy, text, prompt_text, prompt_wav_processed, original_duration,
                                        target_language, base_instruct_command, final_speed_ratio):
    """
    길이를 고려한 스마트 합성: Zero-shot이 너무 길면 Instruct2로 재합성

    Args:
        cosy: CosyVoice2 모델 인스턴스
        text: 합성할 텍스트
        prompt_text: 프롬프트 텍스트
        prompt_wav_processed: 처리된 프롬프트 오디오
        original_duration: 원본 오디오 길이 (초)
        target_language: 타겟 언어
        base_instruct_command: 기본 instruct 명령어
        final_speed_ratio: 속도 비율

    Returns:
        tuple: (선택된 오디오, 사용된 방법, 실제 길이)
    """

    # 1단계: Zero-shot 합성
    logging.info(f"  → [{target_language}] Zero-shot 합성 시도...")
    results_zero = cosy.inference_zero_shot(
        text,
        prompt_text,
        prompt_wav_processed,
        "",
        stream=True,
        text_frontend=True,
        speed=final_speed_ratio
    )

    if results_zero is None:
        logging.error(f"  ❌ Zero-shot 합성 실패")
        return None, None, 0

    # Zero-shot 결과 처리
    result_list = list(results_zero)
    if not result_list:
        logging.error(f"  ❌ Zero-shot 합성 결과가 비어 있음")
        return None, None, 0

    # 모든 결과 연결
    combined_audio = []
    for out in result_list:
        if 'tts_speech' in out:
            speech = out['tts_speech']
            if speech.device.type != 'cpu':
                speech = speech.cpu()
            combined_audio.append(speech)

    if not combined_audio:
        logging.error(f"  ❌ 유효한 Zero-shot 결과 없음")
        return None, None, 0

    zero_shot_audio = torch.cat(combined_audio, dim=1)
    zero_shot_duration = zero_shot_audio.size(1) / 24000  # 24kHz 기준

    logging.info(f"  → Zero-shot 결과: {zero_shot_duration:.2f}s (원본: {original_duration:.2f}s)")

    # 2단계: 길이 비교 및 판단
    duration_ratio = zero_shot_duration / original_duration
    tolerance = 0.1  # 50% 허용 오차로 증가하여 Instruct2 호출 빈도 대폭 감소

    if duration_ratio <= (1.0 + tolerance):
        # Zero-shot 결과가 적절함
        logging.info(f"  ✅ Zero-shot 길이 적절함 (비율: {duration_ratio:.2f})")

        # 메모리 절약: 즉시 가비지 컬렉션
        import gc
        gc.collect()

        return zero_shot_audio, "zero_shot", zero_shot_duration

    # 3단계: Zero-shot이 너무 길면 Instruct2로 재시도
    logging.info(f"  ⚠️ Zero-shot 너무 김 (비율: {duration_ratio:.2f}) - Instruct2로 재시도")

    # 빠르게 말하기 명령어 생성
    fast_command = get_language_specific_instruct_command("빠르게 말해", target_language)
    logging.info(f"  → 빠르게 말하기 명령어: '{fast_command}'")

    # Instruct2 합성 (더 빠른 속도로)
    faster_speed_ratio = min(final_speed_ratio * 1.2, 2.0)  # 더 빠르게 조정
    logging.info(f"  → Instruct2 속도: {faster_speed_ratio:.2f}배")

    results_instruct = cosy.inference_instruct2(
        text,
        fast_command,
        prompt_wav_processed,
        stream=True,
        speed=faster_speed_ratio
    )

    if results_instruct is None:
        logging.warning(f"  ⚠️ Instruct2 합성 실패 - Zero-shot 결과 사용")
        # 메모리 절약
        import gc
        gc.collect()
        return zero_shot_audio, "zero_shot_fallback", zero_shot_duration

    # Instruct2 결과 처리
    instruct_result_list = list(results_instruct)
    if not instruct_result_list:
        logging.warning(f"  ⚠️ Instruct2 결과 비어있음 - Zero-shot 결과 사용")
        # 메모리 절약
        import gc
        gc.collect()
        return zero_shot_audio, "zero_shot_fallback", zero_shot_duration

    # Instruct2 결과 연결
    instruct_combined_audio = []
    for out in instruct_result_list:
        if 'tts_speech' in out:
            speech = out['tts_speech']
            if speech.device.type != 'cpu':
                speech = speech.cpu()
            instruct_combined_audio.append(speech)

    if not instruct_combined_audio:
        logging.warning(f"  ⚠️ 유효한 Instruct2 결과 없음 - Zero-shot 결과 사용")
        # 메모리 절약
        import gc
        gc.collect()
        return zero_shot_audio, "zero_shot_fallback", zero_shot_duration

    instruct_audio = torch.cat(instruct_combined_audio, dim=1)
    instruct_duration = instruct_audio.size(1) / 24000
    logging.info(f"  → Instruct2 결과: {instruct_duration:.2f}s")

    # 4단계: 더 나은 결과 선택
    if instruct_duration <= zero_shot_duration:
        # Instruct2가 더 나음 - Zero-shot 메모리 해제
        logging.info(f"  ✅ Instruct2 결과 선택 (더 적절한 길이)")
        del zero_shot_audio  # 명시적 메모리 해제
        import gc
        gc.collect()
        return instruct_audio, "instruct2_fast", instruct_duration
    else:
        # Zero-shot이 여전히 나음 - Instruct2 메모리 해제
        logging.info(f"  ✅ Zero-shot 결과 선택 (Instruct2도 길어짐)")
        del instruct_audio  # 명시적 메모리 해제
        import gc
        gc.collect()
        return zero_shot_audio, "zero_shot_final", zero_shot_duration


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


def preprocess_text_for_synthesis(text: str) -> str:
    """
    합성용 텍스트 전처리 - 늘어짐 방지

    Args:
        text: 원본 텍스트

    Returns:
        전처리된 텍스트
    """
    import re

    # 과도한 문장 부호 제거
    text = re.sub(r'[.]{2,}', '.', text)  # 연속 마침표 → 단일 마침표
    text = re.sub(r'[,]{2,}', ',', text)  # 연속 쉼표 → 단일 쉼표
    text = re.sub(r'[!]{2,}', '!', text)  # 연속 느낌표 → 단일 느낌표
    text = re.sub(r'[?]{2,}', '?', text)  # 연속 물음표 → 단일 물음표

    # 과도한 공백 제거
    text = re.sub(r'\s{2,}', ' ', text)  # 연속 공백 → 단일 공백

    # 문장 끝 정리 (자연스러운 종료를 위해)
    text = text.strip()
    if text and not text[-1] in '.!?':
        text += '.'  # 문장 부호가 없으면 마침표 추가

    # 너무 긴 문장 분할 (80자 이상)
    if len(text) > 80:
        # 쉼표나 접속사에서 자연스럽게 분할
        split_points = ['하지만', '그러나', '그런데', '그리고', '또한', '그래서']
        for point in split_points:
            if point in text:
                parts = text.split(point, 1)
                if len(parts) == 2 and len(parts[0]) > 20:
                    text = parts[0].strip() + '.'
                    break

    return text


# 언어별 특성 정의
LANGUAGE_CONFIGS = {
    'english': {
        'code': 'en',
        'name': 'English',
        'voice_style': 'natural',
        'speech_rate': 1.2,
        'phoneme_emphasis': 0.8
    },
    'chinese': {
        'code': 'zh',
        'name': '中文',
        'voice_style': 'natural',
        'speech_rate': 1.2,
        'phoneme_emphasis': 0.8
    },
    'japanese': {
        'code': 'ja',
        'name': '日本語',
        'voice_style': 'natural',
        'speech_rate': 1.2,
        'phoneme_emphasis': 0.8
    },
    'korean': {
        'code': 'ko',
        'name': '한국어',
        'voice_style': 'natural',
        'speech_rate': 1.1,
        'phoneme_emphasis': 1.0
    }
}


def detect_text_language(text: str) -> str:
    """
    텍스트의 언어를 감지합니다.
    """
    import re

    # 영어 문자 비율 계산 (공백 제외)
    non_space_chars = [c for c in text if not c.isspace()]
    if not non_space_chars:
        return 'korean'  # 기본값

    english_chars = len([c for c in non_space_chars if ord(c) < 128])
    korean_chars = len([c for c in non_space_chars if 0xAC00 <= ord(c) <= 0xD7A3])
    chinese_chars = len([c for c in non_space_chars if 0x4E00 <= ord(c) <= 0x9FFF])
    japanese_chars = len([c for c in non_space_chars if (0x3040 <= ord(c) <= 0x309F)])  # 히라가나 + 가타카나

    total_chars = len(non_space_chars)

    # 비율 계산
    english_ratio = english_chars / total_chars
    korean_ratio = korean_chars / total_chars
    chinese_ratio = chinese_chars / total_chars
    japanese_ratio = japanese_chars / total_chars

    logging.debug(
        f"Language detection ratios - EN: {english_ratio:.2f}, KO: {korean_ratio:.2f}, ZH: {chinese_ratio:.2f}, JA: {japanese_ratio:.2f}")

    # 언어 판별 (임계값 기반)
    if english_ratio > 0.7:
        detected_lang = 'english'
    elif korean_ratio > 0.5:
        detected_lang = 'korean'
    elif chinese_ratio > 0.5:
        detected_lang = 'chinese'
    elif japanese_ratio > 0.3:
        detected_lang = 'japanese'
    else:
        # 혼재된 경우, 가장 높은 비율 선택
        ratios = {
            'english': english_ratio,
            'korean': korean_ratio,
            'chinese': chinese_ratio,
            'japanese': japanese_ratio
        }
        detected_lang = max(ratios, key=ratios.get)

    logging.info(f"Detected language: {detected_lang} for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    return detected_lang


def preprocess_text_by_language(text: str, target_language: str) -> str:
    """
    언어별 특성에 맞는 텍스트 전처리
    """
    import re

    # 기본 정리
    text = text.strip()

    if target_language == 'english':
        # 영어: 자연스러운 문장 구조 유지
        text = re.sub(r'\s+', ' ', text)  # 연속 공백 정리
        text = re.sub(r'[.]{2,}', '.', text)  # 연속 마침표 정리

        # 영어 특수 처리: 약어 처리
        text = re.sub(r'\bDr\.\s*', 'Doctor ', text)
        text = re.sub(r'\bMr\.\s*', 'Mister ', text)
        text = re.sub(r'\bMrs\.\s*', 'Misses ', text)

    elif target_language == 'chinese':
        # 중국어: 간체자 우선, 성조 고려
        text = re.sub(r'[，]{2,}', '，', text)  # 연속 쉼표 정리
        text = re.sub(r'[。]{2,}', '。', text)  # 연속 마침표 정리

    elif target_language == 'japanese':
        # 일본어: 높임말 처리, 자연스러운 종결어미
        text = re.sub(r'[、]{2,}', '、', text)  # 연속 독점 정리
        text = re.sub(r'[。]{2,}', '。', text)  # 연속 마침표 정리

    elif target_language == 'korean':
        # 한국어: 존댓말 처리
        text = re.sub(r'[,]{2,}', ',', text)  # 연속 쉼표 정리
        text = re.sub(r'[.]{2,}', '.', text)  # 연속 마침표 정리

    # 공통 처리: 과도한 감탄사 제거
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)

    return text


def get_language_specific_instruct_command(base_command: str, target_language: str) -> str:
    """
    언어별 특성에 맞는 instruct 명령어 생성 (CosyVoice2 논문 기반 확장)
    """
    # CosyVoice2에서 지원하는 더 다양한 속도 및 스타일 제어 명령어
    if target_language == 'english':
        return {
            # 속도 제어
            '자연스럽게 말해': 'Speak naturally with normal pace',
            '활기차게 말해': 'Speak with energy and enthusiasm',
            '차분하게 말해': 'Speak calmly and peacefully',
            '감정적으로 말해': 'Speak with strong emotion',
            '천천히 말해': 'Speak slowly and clearly',
            '빠르게 말해': 'Speak quickly and briskly',
            '매우 천천히 말해': 'Speak very slowly with clear articulation',
            '매우 빠르게 말해': 'Speak very quickly but distinctly',
            # 추가 스타일 제어 (CosyVoice2 논문 기반)
            '부드럽게 말해': 'Speak softly and gently',
            '힘차게 말해': 'Speak with strong voice and power',
            '속삭이듯 말해': 'Speak in a whisper',
            '또렷하게 말해': 'Speak with clear pronunciation',
            '리듬감 있게 말해': 'Speak with good rhythm and flow'
        }.get(base_command, base_command)

    elif target_language == 'chinese':
        return {
            # 속도 제어
            '자연스럽게 말해': '自然地说话',
            '활기차게 말해': '充满活力地说话',
            '차분하게 말해': '平静地说话',
            '감정적으로 말해': '富有感情地说话',
            '천천히 말해': '慢慢地说话',
            '빠르게 말해': '快速地说话',
            '매우 천천히 말해': '非常慢地说话',
            '매우 빠르게 말해': '非常快地说话',
            # 추가 스타일 제어
            '부드럽게 말해': '温柔地说话',
            '힘차게 말해': '有力地说话',
            '속삭이듯 말해': '轻声说话',
            '또렷하게 말해': '清楚地说话',
            '리듬감 있게 말해': '有节奏地说话'
        }.get(base_command, base_command)

    elif target_language == 'japanese':
        return {
            # 속도 제어
            '자연스럽게 말해': '自然に話してください',
            '활기차게 말해': '元気よく話してください',
            '차분하게 말해': '落ち着いて話してください',
            '감정적으로 말해': '感情豊かに話してください',
            '천천히 말해': 'ゆっくりと話してください',
            '빠르게 말해': '速く話してください',
            '매우 천천히 말해': '非常にゆっくり話してください',
            '매우 빠르게 말해': '非常に速く話してください',
            # 추가 스타일 제어
            '부드럽게 말해': '優しく話してください',
            '힘차게 말해': '力強く話してください',
            '속삭이듯 말해': '囁くように話してください',
            '또렷하게 말해': 'はっきりと話してください',
            '리듬감 있게 말해': 'リズムよく話してください'
        }.get(base_command, base_command)

    else:  # Korean (default)
        return {
            # 속도 제어
            '자연스럽게 말해': '자연스럽게 말해',
            '활기차게 말해': '활기차게 말해',
            '차분하게 말해': '차분하게 말해',
            '감정적으로 말해': '감정적으로 말해',
            '천천히 말해': '천천히 말해',
            '빠르게 말해': '빠르게 말해',
            '매우 천천히 말해': '매우 천천히 말해',
            '매우 빠르게 말해': '매우 빠르게 말해',
            # 추가 스타일 제어 (CosyVoice2 논문 기반)
            '부드럽게 말해': '부드럽게 말해',
            '힘차게 말해': '힘차게 말해',
            '속삭이듯 말해': '속삭이듯 말해',
            '또렷하게 말해': '또렷하게 말해',
            '리듬감 있게 말해': '리듬감 있게 말해'
        }.get(base_command, base_command)


# 배치 합성 함수
def main(audio_dir, prompt_text_dir, text_dir, out_dir, model_path=LOCAL_COSYVOICE_MODEL, enable_instruct=True,
         manual_command=None, target_language=None):
    # Device 설정 (MPS 지원 제외)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("CUDA GPU 가속 사용 중")
    else:
        device = torch.device("cpu")
        logging.info("CPU 사용 중")

    # 입력 디렉토리 존재 여부 확인
    missing_dirs = []
    if not os.path.exists(audio_dir):
        missing_dirs.append(f"오디오 디렉토리: {audio_dir}")
    if not os.path.exists(prompt_text_dir):
        missing_dirs.append(f"프롬프트 텍스트 디렉토리: {prompt_text_dir}")
    if not os.path.exists(text_dir):
        missing_dirs.append(f"대상 텍스트 디렉토리: {text_dir}")

    if missing_dirs:
        logging.error("필요한 입력 디렉토리가 존재하지 않습니다:")
        for missing_dir in missing_dirs:
            logging.error(f"   - {missing_dir}")
        logging.error("⚠️ 처리할 파일이 없어 종료합니다.")
        return

    # 타겟 언어 자동 감지 (경로에서 추출)
    if target_language is None:
        # text_dir 경로에서 언어 추출 (예: .../english/free/)
        path_parts = text_dir.split(os.sep)
        for part in path_parts:
            if part.lower() in LANGUAGE_CONFIGS:
                target_language = part.lower()
                break

        if target_language is None:
            target_language = 'korean'  # 기본값

    logging.info(f"Target language detected/set: {target_language}")

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
    try:
        audio_files = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')])
        prompt_files = sorted([f for f in os.listdir(prompt_text_dir) if f.lower().endswith('.txt')])
        text_files = sorted([f for f in os.listdir(text_dir) if f.lower().endswith('.txt')])
    except Exception as e:
        logging.error(f"입력 파일 목록을 읽는 데 실패: {e}")
        return

    logging.info(
        f"[디버그] 오디오 파일 ({len(audio_files)}): {audio_files[:5]}{'...' if len(audio_files) > 5 else ''}")
    logging.info(
        f"[디버그] 프롬프트 텍스트 파일 ({len(prompt_files)}): {prompt_files[:5]}{'...' if len(prompt_files) > 5 else ''}")
    logging.info(
        f"[디버그] 대상 텍스트 파일 ({len(text_files)}): {text_files[:5]}{'...' if len(text_files) > 5 else ''}")

    # 파일 매칭 개선: 실제 파일명 패턴에 맞게 매칭
    matched_files = []

    for i, audio_file in enumerate(audio_files):
        # 오디오 파일명에서 기본 이름 추출 (예: vocal_video22_extracted.wav_10_001.wav -> vocal_video22_extracted.wav_10_001)
        audio_base = os.path.splitext(audio_file)[0]

        # 안전한 파일명 분할 처리
        try:
            if '_' in audio_base:
                # "조용석_1m_001"에서 "조용석_1m"와 "001" 분리
                parts = audio_base.rsplit('_', 1)
                if len(parts) >= 2 and parts[-1].isdigit():
                    audio_base = '_'.join(parts[:-1])  # 마지막 숫자 부분 제외한 모든 부분
                    segment_num = parts[-1]  # 마지막 숫자 부분
                else:
                    segment_num = f"{i:03d}"
            else:
                segment_num = f"{i:03d}"
        except Exception as e:
            logging.warning(f"파일명 처리 오류 ({audio_file}): {e}, 기본값 사용")
            segment_num = f"{i:03d}"

        # 프롬프트 텍스트 파일 찾기 (예: vocal_video22_extracted.wav_10_001.ko.txt)
        prompt_file = f"{audio_base}_{segment_num}.ko.txt"
        prompt_file_path = os.path.join(prompt_text_dir, prompt_file)

        # 대상 텍스트 파일 찾기 (예: vocal_video22_extracted.wav_10_001.ko.txt)
        target_file = f"{audio_base}_{segment_num}.ko.txt"
        target_file_path = os.path.join(text_dir, target_file)

        # 파일 존재 여부 확인
        if os.path.exists(prompt_file_path) and os.path.exists(target_file_path):
            matched_files.append((audio_file, prompt_file, target_file))
        else:
            missing_files = []
            if not os.path.exists(prompt_file_path):
                missing_files.append(f"프롬프트({prompt_file})")
            if not os.path.exists(target_file_path):
                missing_files.append(f"대상({target_file})")
            logging.warning(f"[건너뜀] {audio_file} - 누락된 파일: {', '.join(missing_files)}")

    logging.info(f"[정보] 매칭된 파일 세트: {len(matched_files)} / {len(audio_files)} 개")

    if len(matched_files) == 0:
        logging.error("❌ 매칭된 파일이 없습니다. 파일명 패턴을 확인해주세요.")
        return

    # 출력 디렉토리 강제 생성
    zero_shot_dir = os.path.join(out_dir, 'zero_shot')
    instruct_dir = os.path.join(out_dir, 'instruct')

    try:
        os.makedirs(zero_shot_dir, exist_ok=True)
        os.makedirs(instruct_dir, exist_ok=True)
        logging.info(f"✅ 출력 디렉토리 생성 완료:")
        logging.info(f"   - Zero-shot: {zero_shot_dir}")
        logging.info(f"   - Instruct: {instruct_dir}")
    except Exception as e:
        logging.error(f"❌ 출력 디렉토리 생성 실패: {e}")
        return

    current_seed = random.randint(0, 2**32 - 1)
    random.seed(current_seed)
    np.random.seed(current_seed)
    torch.manual_seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)
    logging.info(f"랜덤 시드 사용 중: {current_seed} / 디바이스: {device}")

    # 파일별 합성
    for i, (awav, ptxt, txt) in enumerate(matched_files, 1):
        # 파일 경로 안전화
        safe_awav = sanitize_filename(awav)
        safe_ptxt = sanitize_filename(ptxt)
        safe_txt = sanitize_filename(txt)

        wav_path = safe_file_operations(os.path.join(audio_dir, awav), "read")
        ptxt_path = safe_file_operations(os.path.join(prompt_text_dir, ptxt), "read")
        txt_path = safe_file_operations(os.path.join(text_dir, txt), "read")

        # 파일 경로 오류 체크
        if wav_path.startswith("❌") or ptxt_path.startswith("❌") or txt_path.startswith("❌"):
            logging.error(f"  ❌ 파일 경로 오류:")
            if wav_path.startswith("❌"):
                logging.error(f"    - {wav_path}")
            if ptxt_path.startswith("❌"):
                logging.error(f"    - {ptxt_path}")
            if txt_path.startswith("❌"):
                logging.error(f"    - {txt_path}")
            continue

        # 로깅 (안전화된 파일명 표시)
        if safe_awav != awav or safe_ptxt != ptxt or safe_txt != txt:
            logging.info(f"[{i}/{len(matched_files)}] 처리 중 (파일명 안전화됨)")
            logging.info(f"  → 오디오: {awav} → {safe_awav}")
            logging.info(f"  → 프롬프트: {ptxt} → {safe_ptxt}")
            logging.info(f"  → 텍스트: {txt} → {safe_txt}")
        else:
            logging.info(f"[{i}/{len(matched_files)}] 처리 중 → {awav} / {ptxt} / {txt}")

        # 파일 존재 여부 확인 및 로깅
        missing_files = []
        if not os.path.exists(wav_path):
            missing_files.append(f"오디오: {wav_path}")
        if not os.path.exists(ptxt_path):
            missing_files.append(f"프롬프트 텍스트: {ptxt_path}")
        if not os.path.exists(txt_path):
            missing_files.append(f"대상 텍스트: {txt_path}")

        if missing_files:
            logging.error(f"  ❌ 누락된 파일: {', '.join(missing_files)}")
            continue

        # 오디오 & 텍스트 로드
        try:
            prompt_wav = load_wav_resample(wav_path)
        except Exception as e:
            logging.error(f"  ❌ 오디오 로드 실패 ({wav_path}): {e}")
            continue

        # 원본 오디오 길이 추적 (패딩 없이 정확한 길이)
        try:
            original_wav = load_wav_resample(wav_path, min_duration=0.0)  # 패딩 없이 로드
            original_duration = original_wav.size(1) / 16000  # 초 단위
        except Exception as e:
            logging.error(f"  ❌ 원본 오디오 길이 측정 실패: {e}")
            continue

        try:
            with open(ptxt_path, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            logging.error(f"  ❌ 텍스트 파일 읽기 실패: {e}")
            continue

        # 텍스트 유효성 검사
        if not prompt_text or len(prompt_text.strip()) == 0:
            logging.error(f"  ❌ 프롬프트 텍스트가 비어 있습니다: {ptxt_path}")
            continue
        if not text or len(text.strip()) == 0:
            logging.error(f"  ❌ 대상 텍스트가 비어 있습니다: {txt_path}")
            continue

        # 텍스트 전처리 추가 (늘어짐 방지)
        original_text = text
        original_prompt_text = prompt_text
        text = preprocess_text_for_synthesis(text)
        prompt_text = preprocess_text_for_synthesis(prompt_text)

        # 전처리 결과 로깅
        if text != original_text:
            logging.info(f"  → 텍스트 전처리: '{original_text[:30]}...' → '{text[:30]}...'")
        if prompt_text != original_prompt_text:
            logging.info(f"  → 프롬프트 텍스트 전처리: '{original_prompt_text[:20]}...' → '{prompt_text[:20]}...'")

        # 파일명에서 기본 이름과 세그먼트 번호 추출
        base_name = os.path.splitext(awav)[0]  # 예: "조용석_1m_001"
        if '_' in base_name:
            # "조용석_1m_001"에서 "조용석_1m"와 "001" 분리
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

        logging.info(f"  → [{target_language}] 프롬프트 오디오 길이: {prompt_wav.size(1) / 16000:.2f}s")

        # 합성 전 필수 조건 재확인
        if prompt_wav is None or prompt_wav.size(1) == 0:
            logging.error(f"  ❌ 처리된 프롬프트 오디오가 비어 있습니다")
            continue
        if not text.strip() or not prompt_text.strip():
            logging.error(f"  ❌ 처리된 텍스트가 비어 있습니다")
            continue

        # 언어별 속도 조정 적용
        lang_config = LANGUAGE_CONFIGS.get(target_language, LANGUAGE_CONFIGS['korean'])
        base_speed_ratio = lang_config['speech_rate']

        # Zero-shot 및 Instruct2 합성
        audio_result, method_used, duration = smart_synthesis_with_length_control(
            cosy,
            text,
            prompt_text,
            prompt_wav,
            original_duration,
            target_language,
            manual_command if manual_command else '자연스럽게 말해',
            base_speed_ratio
        )

        if audio_result is not None:
            logging.info(f"  ✅ [{target_language}] 최종 결과: {duration:.2f}s (방법: {method_used})")
        else:
            logging.error(f"  ❌ [{target_language}] 합성 결과 없음")
            continue

        # 결과 저장
        if method_used == "instruct2_fast":
            try:
                logging.info(f"  → [{target_language}] Instruct2 결과 저장 시작...")

                # Instruct2 디렉토리 확실히 생성
                if not os.path.exists(instruct_dir):
                    os.makedirs(instruct_dir, exist_ok=True)
                    logging.info(f"  → Instruct2 출력 디렉토리 생성: {instruct_dir}")

                # 안전한 파일명 생성
                safe_name = sanitize_filename(f"{audio_base}_{segment_num}_instruct.wav")
                save_path = os.path.join(instruct_dir, safe_name)

                try:
                    torchaudio.save(save_path, audio_result, 24000)
                    final_duration = audio_result.size(1) / 24000

                    # 파일 저장 확인
                    if os.path.exists(save_path):
                        file_size = os.path.getsize(save_path)
                        logging.info(
                            f"    ✅ Instruct2 저장 완료 ➜ {safe_name} ({final_duration:.2f}초, {file_size} 바이트)")
                    else:
                        logging.error(f"    ❌ 파일이 저장되지 않았습니다: {save_path}")

                except Exception as save_error:
                    logging.error(f"    ❌ Instruct2 파일 저장 실패: {save_error}")
            except Exception as e:
                logging.error(f"    ❌ [{target_language}] Instruct2 처리 실패: {e}")
                import traceback
                logging.error(f"    상세 오류: {traceback.format_exc()}")
        else:
            # Zero-shot 결과 저장
            try:
                logging.info(f"  → [{target_language}] Zero-shot 결과 저장 시작...")

                # 안전한 파일명 생성
                safe_name = sanitize_filename(f"{audio_base}_{segment_num}.wav")
                save_path = os.path.join(zero_shot_dir, safe_name)

                # 디렉토리 확인 및 생성
                if not os.path.exists(zero_shot_dir):
                    os.makedirs(zero_shot_dir, exist_ok=True)

                try:
                    torchaudio.save(save_path, audio_result, 24000)
                    final_duration = audio_result.size(1) / 24000

                    # 파일 저장 확인
                    if os.path.exists(save_path):
                        file_size = os.path.getsize(save_path)
                        logging.info(
                            f"    ✅ Zero-shot 저장 완료 ➜ {safe_name} ({final_duration:.2f}초, {file_size} 바이트)")
                    else:
                        logging.error(f"    ❌ 파일이 저장되지 않았습니다: {save_path}")

                except Exception as save_error:
                    logging.error(f"    ❌ torchaudio.save 실패: {save_error}")
            except Exception as e:
                logging.error(f"  ❌ [{target_language}] Zero-shot 저장 실패: {e}")
                import traceback
                logging.error(f"  상세 오류: {traceback.format_exc()}")

        # 메모리 정리 (각 파일 처리 후)
        cleanup_memory(device)

    # CosyVoice 모델 메모리 해제
    cleanup_cosyvoice_model()

    logging.info(f"✅ [{target_language}] 배치 처리 완료 - {len(matched_files)} 개 파일 처리됨")


# 스크립트 엔트리포인트
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="배치 Zero-shot TTS with CosyVoice2 (Language-aware)"
    )
    parser.add_argument('--audio_dir', required=True, help="참조 .wav 폴더 경로")
    parser.add_argument('--prompt_text_dir', required=True, help="프롬프트 텍스트 (.txt) 폴더 경로")
    parser.add_argument('--text_dir', required=True, help="합성 텍스트 (.txt) 폴더 경로")
    parser.add_argument('--out_dir', required=True, help="출력 WAV 폴더 경로")
    parser.add_argument('--model_path', default=LOCAL_COSYVOICE_MODEL, help="CosyVoice2 모델 경로")
    parser.add_argument('--enable_instruct', action='store_true', default=False, help="Instruct2 기능 활성화")
    parser.add_argument('--manual_command', type=str, default=None, help="수동 지정 instruct 명령어")
    parser.add_argument('--target_language', type=str, default=None, help="타겟 언어 (english/chinese/japanese/korean)")
    args = parser.parse_args()

    main(
        args.audio_dir,
        args.prompt_text_dir,
        args.text_dir,
        args.out_dir,
        model_path=args.model_path,
        enable_instruct=args.enable_instruct,
        manual_command=args.manual_command,
        target_language=args.target_language
    )
