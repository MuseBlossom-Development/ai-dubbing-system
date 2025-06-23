#!/usr/bin/env python3
"""
Instruct2 저장 실패 원인을 찾기 위한 테스트 스크립트
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def test_instruct_save():
    """Instruct2 저장 테스트"""

    # 테스트 디렉토리 설정
    test_dir = "split_audio/vocal_video22_extracted.wav_10"
    audio_dir = os.path.join(test_dir, "wav")
    prompt_text_dir = os.path.join(test_dir, "txt", "ko")
    text_dir = os.path.join(test_dir, "txt", "english", "free")

    print("🔍 디렉토리 확인:")
    print(f"  - 오디오: {os.path.exists(audio_dir)} ({audio_dir})")
    print(f"  - 프롬프트: {os.path.exists(prompt_text_dir)} ({prompt_text_dir})")
    print(f"  - 대상텍스트: {os.path.exists(text_dir)} ({text_dir})")

    if not all([os.path.exists(audio_dir), os.path.exists(prompt_text_dir), os.path.exists(text_dir)]):
        print("❌ 필요한 디렉토리가 없습니다")
        return

    # 파일 찾기
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if not audio_files:
        print("❌ 오디오 파일이 없습니다")
        return

    # 첫 번째 파일로 테스트
    test_audio = audio_files[0]
    audio_base = os.path.splitext(test_audio)[0]

    audio_path = os.path.join(audio_dir, test_audio)
    prompt_path = os.path.join(prompt_text_dir, f"{audio_base}.ko.txt")
    target_path = os.path.join(text_dir, f"{audio_base}.ko.txt")

    print(f"🧪 테스트 파일:")
    print(f"  - 오디오: {audio_path} ({'존재' if os.path.exists(audio_path) else '없음'})")
    print(f"  - 프롬프트: {prompt_path} ({'존재' if os.path.exists(prompt_path) else '없음'})")
    print(f"  - 대상: {target_path} ({'존재' if os.path.exists(target_path) else '없음'})")

    if not all([os.path.exists(audio_path), os.path.exists(prompt_path), os.path.exists(target_path)]):
        print("❌ 테스트 파일이 없습니다")
        return

    # 텍스트 읽기
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        with open(target_path, 'r', encoding='utf-8') as f:
            target_text = f.read().strip()

        print(f"📝 텍스트:")
        print(f"  - 프롬프트: '{prompt_text[:30]}...'")
        print(f"  - 대상: '{target_text[:30]}...'")

    except Exception as e:
        print(f"❌ 텍스트 읽기 실패: {e}")
        return

    # CosyVoice2 모델 로드
    try:
        print("🤖 CosyVoice2 모델 로드 중...")
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from batch_cosy import load_wav_resample, postprocess

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
        return

    # 오디오 전처리
    try:
        print("🎵 오디오 전처리 중...")
        prompt_wav = load_wav_resample(audio_path)
        prompt_wav_processed = postprocess(prompt_wav)
        print(f"  - 전처리 완료: {prompt_wav_processed.size(1) / 16000:.2f}초")

    except Exception as e:
        print(f"❌ 오디오 전처리 실패: {e}")
        return

    # 출력 디렉토리 생성
    test_output_dir = "test_instruct_output"
    os.makedirs(test_output_dir, exist_ok=True)
    print(f"📁 출력 디렉토리: {test_output_dir}")

    # Instruct2 합성 테스트
    try:
        print("🔊 Instruct2 합성 테스트 중...")

        instruct_command = "자연스럽게 말해"
        print(f"  - 명령어: {instruct_command}")

        results = cosy.inference_instruct2(
            target_text,
            instruct_command,
            prompt_wav_processed,
            stream=False,
            speed=1.0
        )

        print(f"  - 합성 결과 타입: {type(results)}")

        # 결과 처리
        if results is None:
            print("❌ 합성 결과가 None입니다")
            return

        result_list = list(results)
        print(f"  - 결과 개수: {len(result_list)}")

        if not result_list:
            print("❌ 합성 결과가 비어있습니다")
            return

        # 첫 번째 결과 저장 테스트
        first_result = result_list[0]
        print(f"  - 첫 번째 결과 키들: {first_result.keys()}")

        if 'tts_speech' not in first_result:
            print("❌ 'tts_speech' 키가 없습니다")
            return

        speech = first_result['tts_speech']
        print(f"  - 스피치 텐서 크기: {speech.shape}")
        print(f"  - 스피치 디바이스: {speech.device}")

        # CPU로 이동
        if speech.device.type != 'cpu':
            speech = speech.cpu()
            print("  - CPU로 이동 완료")

        # 저장 테스트
        test_file = os.path.join(test_output_dir, "test_instruct2_output.wav")
        print(f"  - 저장 경로: {test_file}")

        torchaudio.save(test_file, speech, 24000)

        # 저장 확인
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            duration = speech.size(1) / 24000
            print(f"✅ 저장 성공!")
            print(f"  - 파일 크기: {file_size} bytes")
            print(f"  - 길이: {duration:.2f}초")
        else:
            print("❌ 파일이 저장되지 않았습니다")

    except Exception as e:
        print(f"❌ Instruct2 합성 실패: {e}")
        import traceback
        print(f"상세 오류: {traceback.format_exc()}")


if __name__ == "__main__":
    print("🚀 Instruct2 저장 테스트 시작")
    test_instruct_save()
    print("🏁 테스트 완료")
