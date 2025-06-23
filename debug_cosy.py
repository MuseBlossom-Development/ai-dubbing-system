#!/usr/bin/env python3
"""
CosyVoice2 음성 합성 디버그 스크립트
실패 원인을 파악하기 위한 단일 파일 테스트
"""

import os
import sys
import logging
import torch
import torchaudio

# 프로젝트 내 CosyVoice2 모델 로컬 경로 설정
repo_root = os.path.dirname(__file__)
LOCAL_COSYVOICE_MODEL = os.path.join(
    repo_root, 'CosyVoice', 'pretrained_models', 'CosyVoice2-0.5B'
)

# CosyVoice2 패키지 경로 추가
sys.path.insert(0, os.path.join(repo_root, 'CosyVoice'))

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


def debug_single_synthesis():
    """단일 파일로 합성 테스트"""

    # 테스트 파일 경로 설정
    test_dir = "split_audio/vocal_video22_extracted.wav_10"
    audio_dir = os.path.join(test_dir, "wav")
    prompt_text_dir = os.path.join(test_dir, "txt", "ko")
    text_dir = os.path.join(test_dir, "txt", "english", "free")

    # 디렉토리 존재 여부 확인
    print(f"🔍 디렉토리 존재 여부 확인:")
    print(f"  - 오디오 디렉토리: {os.path.exists(audio_dir)} ({audio_dir})")
    print(f"  - 프롬프트 디렉토리: {os.path.exists(prompt_text_dir)} ({prompt_text_dir})")
    print(f"  - 대상 텍스트 디렉토리: {os.path.exists(text_dir)} ({text_dir})")

    # 파일 목록 확인
    audio_files = []
    prompt_files = []
    text_files = []

    if os.path.exists(audio_dir):
        audio_files = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')])
    if os.path.exists(prompt_text_dir):
        prompt_files = sorted([f for f in os.listdir(prompt_text_dir) if f.lower().endswith('.txt')])
    if os.path.exists(text_dir):
        text_files = sorted([f for f in os.listdir(text_dir) if f.lower().endswith('.txt')])

    print(f"🔍 파일 개수 확인:")
    print(f"  - 오디오: {len(audio_files)}")
    print(f"  - 프롬프트 텍스트: {len(prompt_files)}")
    print(f"  - 대상 텍스트: {len(text_files)}")

    if len(audio_files) == 0:
        print("❌ 오디오 파일이 없습니다")
        print("📂 사용가능한 wav 파일들을 찾아보겠습니다...")

        # 대안 경로 탐색
        for root, dirs, files in os.walk("."):
            wav_files = [f for f in files if f.lower().endswith('.wav') and 'video22' in f]
            if wav_files:
                print(f"  🎵 {root}: {len(wav_files)}개 파일")
                for wav_file in wav_files[:3]:  # 최대 3개만 표시
                    print(f"    - {wav_file}")
        return

    # 첫 번째 파일로 테스트
    test_audio = audio_files[0]
    test_prompt_txt = test_audio.replace('.wav', '.txt')
    test_target_txt = test_audio.replace('.wav', '.txt')

    audio_path = os.path.join(audio_dir, test_audio)
    prompt_path = os.path.join(prompt_text_dir, test_prompt_txt)
    target_path = os.path.join(text_dir, test_target_txt)

    print(f"🧪 테스트 파일:")
    print(f"  - 오디오: {audio_path}")
    print(f"  - 프롬프트: {prompt_path}")
    print(f"  - 대상: {target_path}")

    # 파일 존재 여부 확인
    missing = []
    if not os.path.exists(audio_path):
        missing.append("오디오")
    if not os.path.exists(prompt_path):
        missing.append("프롬프트")
    if not os.path.exists(target_path):
        missing.append("대상 텍스트")

    if missing:
        print(f"❌ 누락된 파일: {', '.join(missing)}")

        # 대안 텍스트 파일 찾기
        if "프롬프트" in missing:
            print("🔍 한국어 텍스트 파일들을 찾아보겠습니다...")
            for root, dirs, files in os.walk(test_dir):
                if 'ko' in root or 'korean' in root.lower():
                    txt_files = [f for f in files if f.lower().endswith('.txt')]
                    if txt_files:
                        print(f"  📝 {root}: {len(txt_files)}개 텍스트 파일")

        if "대상 텍스트" in missing:
            print("🔍 영어 텍스트 파일들을 찾아보겠습니다...")
            for root, dirs, files in os.walk(test_dir):
                if 'english' in root.lower() or 'en' in root:
                    txt_files = [f for f in files if f.lower().endswith('.txt')]
                    if txt_files:
                        print(f"  📝 {root}: {len(txt_files)}개 텍스트 파일")
        return

    # 텍스트 내용 확인
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        with open(target_path, 'r', encoding='utf-8') as f:
            target_text = f.read().strip()

        print(f"📝 텍스트 내용:")
        print(f"  - 프롬프트: '{prompt_text[:50]}...'")
        print(f"  - 대상: '{target_text[:50]}...'")

        if not prompt_text:
            print("❌ 프롬프트 텍스트가 비어있습니다")
            return
        if not target_text:
            print("❌ 대상 텍스트가 비어있습니다")
            return

    except Exception as e:
        print(f"❌ 텍스트 파일 읽기 실패: {e}")
        return

    # 오디오 파일 확인
    try:
        waveform, sr = torchaudio.load(audio_path)
        duration = waveform.size(1) / sr
        print(f"🎵 오디오 정보:")
        print(f"  - 샘플레이트: {sr}Hz")
        print(f"  - 채널: {waveform.size(0)}")
        print(f"  - 길이: {duration:.2f}초")

        if duration < 0.5:
            print("⚠️ 오디오가 너무 짧습니다 (0.5초 미만)")
        if duration > 30:
            print("⚠️ 오디오가 너무 깁니다 (30초 초과)")

    except Exception as e:
        print(f"❌ 오디오 파일 읽기 실패: {e}")
        return

    # CosyVoice2 모델 로드 시도
    try:
        print("🤖 CosyVoice2 모델 로드 중...")
        from cosyvoice.cli.cosyvoice import CosyVoice2

        # Device 설정
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("  - GPU 사용")
        else:
            device = torch.device("cpu")
            print("  - CPU 사용")

        cosy = CosyVoice2(
            LOCAL_COSYVOICE_MODEL,
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False
        )
        print("✅ 모델 로드 성공")

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")
        return

    # 실제 합성 테스트
    try:
        print("🔊 합성 테스트 중...")

        # 오디오 전처리
        from batch_cosy import load_wav_resample, postprocess

        prompt_wav = load_wav_resample(audio_path)
        prompt_wav_processed = postprocess(prompt_wav)

        print(f"  - 전처리된 오디오 길이: {prompt_wav_processed.size(1) / 16000:.2f}초")

        # Zero-shot 합성 시도
        results = cosy.inference_zero_shot(
            target_text,
            prompt_text,
            prompt_wav_processed,
            "",
            stream=False,
            text_frontend=True,
            speed=1.0
        )

        # 결과 확인
        result_list = list(results)
        if result_list:
            speech = result_list[0]['tts_speech']
            output_duration = speech.size(1) / 24000
            print(f"✅ 합성 성공! 출력 길이: {output_duration:.2f}초")

            # 테스트 파일 저장
            test_output = "debug_synthesis_test.wav"
            torchaudio.save(test_output, speech.cpu(), 24000)
            print(f"💾 테스트 파일 저장: {test_output}")
        else:
            print("❌ 합성 결과가 비어있습니다")

    except Exception as e:
        print(f"❌ 합성 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")


if __name__ == "__main__":
    print("🚀 CosyVoice2 디버그 테스트 시작")
    debug_single_synthesis()
    print("🏁 테스트 완료")
