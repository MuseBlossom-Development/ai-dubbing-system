# test_multi_translate.py
# 다국어 번역 테스트 스크립트

import os
from gtranslate import (
    translate_to_english,
    translate_to_chinese,
    translate_to_japanese,
    SUPPORTED_LANGUAGES
)


def test_translations():
    """여러 언어로 번역 테스트"""

    # 테스트용 한국어 텍스트
    test_texts = [
        "안녕하세요! 반갑습니다.",
        "오늘 날씨가 정말 좋네요.",
        "저는 한국에서 왔습니다.",
        "맛있는 음식을 먹고 싶어요.",
        "감사합니다. 잘 부탁드립니다."
    ]

    print("=" * 60)
    print("다국어 번역 테스트")
    print("=" * 60)

    for i, korean_text in enumerate(test_texts, 1):
        print(f"\n[{i}] 원본 (한국어): {korean_text}")
        print("-" * 50)

        # 영어 번역
        english_literal = translate_to_english(korean_text, "literal")
        english_free = translate_to_english(korean_text, "free")
        print(f"🇺🇸 영어 (직역): {english_literal}")
        print(f"🇺🇸 영어 (의역): {english_free}")

        # 중국어 번역
        chinese_literal = translate_to_chinese(korean_text, "literal")
        chinese_free = translate_to_chinese(korean_text, "free")
        print(f"🇨🇳 중국어 (직역): {chinese_literal}")
        print(f"🇨🇳 중국어 (의역): {chinese_free}")

        # 일본어 번역
        japanese_literal = translate_to_japanese(korean_text, "literal")
        japanese_free = translate_to_japanese(korean_text, "free")
        print(f"🇯🇵 일본어 (직역): {japanese_literal}")
        print(f"🇯🇵 일본어 (의역): {japanese_free}")


def create_test_files():
    """테스트용 파일 생성"""
    test_dir = "test_translation"
    os.makedirs(test_dir, exist_ok=True)

    # 테스트용 한국어 텍스트 파일들 생성
    test_texts = {
        "greeting.txt": "안녕하세요! 만나서 반갑습니다.",
        "weather.txt": "오늘 날씨가 정말 좋네요.",
        "introduction.txt": "저는 한국에서 온 개발자입니다.",
        "food.txt": "한국 음식이 정말 맛있어요.",
        "thanks.txt": "감사합니다. 잘 부탁드립니다."
    }

    for filename, content in test_texts.items():
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"테스트 파일들이 '{test_dir}' 폴더에 생성되었습니다:")
    for filename in test_texts.keys():
        print(f"  - {filename}")

    return test_dir


def show_usage():
    """사용법 안내"""
    print("\n" + "=" * 60)
    print("다국어 번역 시스템 사용법")
    print("=" * 60)

    print("\n1. 단일 언어 번역:")
    print("   python batch_translate.py input_dir output_dir 0.8 english")
    print("   python batch_translate.py input_dir output_dir 0.8 chinese")
    print("   python batch_translate.py input_dir output_dir 0.8 japanese")

    print("\n2. 여러 언어 동시 번역:")
    print("   python batch_translate.py input_dir output_dir 0.8 english,chinese")
    print("   python batch_translate.py input_dir output_dir 0.8 english,japanese")
    print("   python batch_translate.py input_dir output_dir 0.8 chinese,japanese")

    print("\n3. 모든 언어 번역:")
    print("   python batch_translate.py input_dir output_dir 0.8 all")

    print(f"\n지원 언어: {', '.join(SUPPORTED_LANGUAGES.keys())}")

    print("\n4. 음성 생성:")
    print("   번역 후 생성된 텍스트 파일들을 batch_cosy.py로 음성 생성")
    print(
        "   python batch_cosy.py --audio_dir ref_audio --prompt_text_dir prompt_texts --text_dir translated_texts --out_dir output_audio")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create_files":
        # 테스트 파일 생성
        test_dir = create_test_files()
        print(f"\n다음 명령어로 번역을 테스트해보세요:")
        print(f"python batch_translate.py {test_dir} translation_output 0.8 all")
    elif len(sys.argv) > 1 and sys.argv[1] == "usage":
        # 사용법 표시
        show_usage()
    else:
        # 번역 테스트 실행
        test_translations()
        show_usage()
