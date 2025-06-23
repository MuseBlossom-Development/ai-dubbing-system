# gtranslate.py

import os
import platform
import random
import re
import logging
import subprocess
from llama_cpp import Llama


def get_gpu_memory_usage():
    """GPU 메모리 사용량을 MB 단위로 반환 (Linux/macOS/Windows 모두 지원)"""
    try:
        # 시스템에 따라 다른 명령어 사용
        if platform.system() == "Linux":
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                encoding='utf-8'
            )
        elif platform.system() == "Windows":
            result = subprocess.check_output(
                ["nvidia-smi", "-q", "-d", "MEMORY"],
                encoding='utf-8'
            )
            # Windows에서는 전체 메모리 사용량 추출
            match = re.search(r'Used GPU Memory:\s+(\d+)\s+MiB', result)
            if match:
                return int(match.group(1))
            return 0

        # Linux/macOS 결과 처리
        usage = int(result.strip())
        return usage

    except Exception as e:
        print(f"GPU 메모리 사용량 확인 실패: {e}")
        return -1


# Numba 디버그 로그 억제
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.core').setLevel(logging.WARNING)
logging.getLogger('numba.typed').setLevel(logging.WARNING)

# 모델 파일 경로
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "gemma", "gemma-3-27b-it-q4_0.gguf")

# 전역 LLM 인스턴스
_llm = None
MAX_ATTEMPTS = 5  # 재시도 횟수 증가


def _get_library_path():
    """llama-cpp-python 빌드 후 라이브러리 경로 반환"""
    system = platform.system()
    base_path = os.path.dirname(__file__)

    if system == "Windows":
        # Windows: llama-cpp-python/build/lib/Release/llama_cpp.dll
        return os.path.join(base_path, "llama-cpp-python", "build", "lib", "Release", "llama_cpp.dll")
    elif system == "Darwin":  # macOS
        # macOS: llama-cpp-python/build/lib/libllama_cpp.dylib
        return os.path.join(base_path, "llama-cpp-python", "build", "lib", "libllama_cpp.dylib")
    else:  # Linux
        # Linux: llama-cpp-python/build/lib/libllama_cpp.so
        return os.path.join(base_path, "llama-cpp-python", "build", "lib", "libllama_cpp.so")


def _get_llm():
    global _llm
    if _llm is None:
        # 빌드된 라이브러리 경로 확인
        library_path = _get_library_path()

        # 라이브러리 파일 존재 확인
        if not os.path.exists(library_path):
            print(f"Warning: Built library not found at {library_path}")
            print("🔧 시스템 기본 라이브러리 사용 (pip 설치)")
            library_path = None  # None으로 설정하면 시스템 기본 라이브러리 사용
        else:
            print(f"✅ 사용자 빌드 라이브러리 발견: {library_path}")

        print("🔄 LLM 초기화 중...")
        print(f"📁 모델 경로: {MODEL_PATH}")
        print(f"🔧 라이브러리: {'사용자 빌드' if library_path else '시스템 기본 (pip)'}")

        try:
            _llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=4096,
                n_threads=4,  # GPU 사용 시 스레드 수 줄이기
                n_gpu_layers=-1,  # 모든 레이어를 GPU에서 실행
                n_batch=512,  # 배치 크기 조정
                seed=None,  # 매번 다른 시드 사용하여 컨텍스트 초기화
                library=library_path,
                verbose=True  # GPU 사용 여부 확인을 위해 로그 활성화
            )

            # GPU 사용 여부 확인
            print("✅ LLM 초기화 완료!")

            # GPU 레이어 정보 확인
            try:
                if hasattr(_llm, '_model'):
                    print(f"🎯 GPU 레이어 설정: -1 (모든 레이어)")
                    print("🔍 GPU 사용 여부는 초기화 로그를 확인하세요")
                    print("   - 'using CUDA for GPU acceleration' 메시지 찾기")
                    print("   - 'VRAM used: XXX MB' 메시지 찾기")
            except Exception as e:
                print(f"⚠️ GPU 정보 확인 중 오류: {e}")

        except Exception as e:
            print(f"❌ LLM 초기화 실패: {e}")
            print("💡 CUDA 지원 llama-cpp-python 빌드 필요:")
            print("   cd llama-cpp-python")
            print("   CMAKE_ARGS=\"-DLLAMA_CUDA=on\" pip install -e . --verbose")
            raise

    return _llm


def _reset_llm_context():
    """LLM 컨텍스트를 초기화하여 이전 번역의 영향을 제거"""
    global _llm
    if _llm is not None:
        try:
            # llama_cpp에서 컨텍스트 리셋이 가능한지 확인
            if hasattr(_llm, 'reset'):
                _llm.reset()
            else:
                # reset 메소드가 없으면 모델을 재초기화
                print("LLM 컨텍스트 리셋을 위해 모델 재초기화...")
                cleanup_llm()
                _llm = None
        except Exception as e:
            print(f"LLM 컨텍스트 리셋 중 오류: {e}")
            # 오류 발생 시 모델 재초기화
            cleanup_llm()
            _llm = None


def _cleanup(text: str) -> str:
    """번호·괄호·별표 제거, 빈 줄·여백 정리"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out = "\n".join(lines)
    out = re.sub(r'(?m)^\s*\d+\.\s*', "", out)
    out = re.sub(r'\([^)]*\)', "", out)
    out = out.replace("*", "")
    return "\n".join([ln.strip() for ln in out.split("\n") if ln.strip()])


def _contains_korean(text: str) -> bool:
    """텍스트에 한글이 포함되어 있는지 확인하는 강화된 함수"""
    # 한글 자모, 완성형 한글, 한글 호환 자모 모두 검사
    korean_patterns = [
        r'[가-힣]',  # 완성형 한글
        r'[ㄱ-ㅎ]',  # 한글 자음
        r'[ㅏ-ㅣ]',  # 한글 모음
        r'[\u3130-\u318F]',  # 한글 호환 자모
        r'[\uAC00-\uD7AF]',  # 한글 음절
        r'[\u1100-\u11FF]',  # 한글 자모 확장-A
        r'[\uA960-\uA97F]',  # 한글 자모 확장-B
    ]

    for pattern in korean_patterns:
        if re.search(pattern, text):
            return True
    return False


def _remove_korean_parts(text: str) -> str:
    """텍스트에서 한글이 포함된 부분을 제거"""
    # 한글이 포함된 단어나 구문을 제거
    words = text.split()
    clean_words = []

    for word in words:
        if not _contains_korean(word):
            clean_words.append(word)

    result = " ".join(clean_words).strip()

    # 완전히 비어있으면 빈 문자열 반환
    if not result or result.isspace():
        return ""

    return result


def _enhanced_fallback_translate(text: str, target_lang: str) -> str:
    """강화된 fallback 번역 함수 (일본어 특별 대응)"""
    # 일본어용 확장된 사전
    korean_japanese_dict = {
        "네": "はい", "안녕하세요": "こんにちは", "감사합니다": "ありがとうございます",
        "저는": "私は", "연기도": "演技も", "하고": "して", "코미디도": "コメディも",
        "하는": "する", "코미디원": "コメディアン", "문상훈": "ムン・サンフン",
        "입니다": "です", "맞습니다": "そうです", "어르신": "お年寄り",
        "저희": "私たち", "상품": "商品", "개발": "開発", "쉽게": "簡単に",
        "이해": "理解", "참여": "参加", "특별히": "特に", "있어요": "あります",
        "그룹": "グループ", "디지털": "デジタル", "투자": "投資", "전문": "専門",
        "계열사": "関連会社", "당연하죠": "当然です", "평생": "一生", "열심히": "一生懸命",
        "모으신": "貯めた", "분들": "方々", "너무": "とても", "속상하시죠": "悔しいでしょう"
    }

    korean_chinese_dict = {
        "네": "是的", "안녕하세요": "你好", "감사합니다": "谢谢",
        "저는": "我", "연기도": "表演也", "하고": "做", "코미디도": "喜剧也",
        "하는": "做", "코미디원": "喜剧演员", "문상훈": "文尚勋",
        "입니다": "是", "맞습니다": "对", "어르신": "老人家",
        "저희": "我们", "상품": "产品", "개발": "开发", "쉽게": "容易",
        "이해": "理解", "참여": "参与", "특별히": "特别", "있어요": "有",
        "그룹": "集团", "디지털": "数字", "투자": "投资", "전문": "专业",
        "계열사": "关联公司", "당연하죠": "当然", "평생": "一生", "열심히": "努力",
        "모으신": "积攒", "분들": "人们", "너무": "太", "속상하시죠": "很难过吧"
    }

    korean_english_dict = {
        "네": "Yes", "안녕하세요": "Hello", "감사합니다": "Thank you",
        "저는": "I am", "연기도": "acting too", "하고": "and", "코미디도": "comedy too",
        "하는": "doing", "코미디원": "comedian", "문상훈": "Moon Sang-hun",
        "입니다": "am", "맞습니다": "correct", "어르신": "elders",
        "저희": "we", "상품": "product", "개발": "developed", "쉽게": "easily",
        "이해": "understand", "참여": "participate", "특별히": "specially", "있어요": "have",
        "그룹": "Group", "디지털": "digital", "투자": "investment", "전문": "specialized",
        "계열사": "affiliate", "당연하죠": "of course", "평생": "lifetime", "열심히": "diligently",
        "모으신": "saved", "분들": "people", "너무": "so", "속상하시죠": "upset"
    }

    # 언어별 사전 선택
    if target_lang == "japanese":
        trans_dict = korean_japanese_dict
        default_response = "申し訳ありませんが、翻訳できませんでした"
    elif target_lang == "chinese":
        trans_dict = korean_chinese_dict
        default_response = "抱歉，无法翻译"
    else:  # english
        trans_dict = korean_english_dict
        default_response = "Translation unavailable"

    try:
        # 문장 단위로 번역 시도
        translated_parts = []

        # 공백으로 단어 분리
        words = text.split()
        for word in words:
            # 특수문자 제거한 clean_word
            clean_word = re.sub(r'[^\w가-힣]', '', word)
            translated = None

            # 사전에서 가장 긴 매치 찾기
            for korean, translation in sorted(trans_dict.items(), key=len, reverse=True):
                if korean in clean_word or clean_word in korean:
                    translated = translation
                    break

            if translated:
                translated_parts.append(translated)
            else:
                # 번역되지 않은 단어는 건너뛰기
                continue

        if translated_parts:
            result = " ".join(translated_parts)

            # 일본어의 경우 문장 구조 개선
            if target_lang == "japanese":
                # 기본적인 일본어 문장 구조 개선
                if "私は" in result and "です" in result:
                    # "私は X です" 형태로 정리
                    result = re.sub(r'私は\s*(.*?)\s*です', r'私は\1です', result)
                # 중복 제거
                result = re.sub(r'\b(\w+)\s+\1\b', r'\1', result)

            return result.strip()

    except Exception as e:
        print(f"강화된 fallback 번역 오류: {e}")

    return default_response


def cleanup_llm():
    """LLM 객체를 메모리에서 해제하는 함수"""
    global _llm

    if _llm is not None:
        try:
            # llama_cpp의 Llama 객체 해제
            if hasattr(_llm, 'close'):
                _llm.close()
            elif hasattr(_llm, '__del__'):
                _llm.__del__()
        except Exception as e:
            print(f"LLM 객체 해제 중 오류: {e}")
        finally:
            _llm = None

    # 가비지 콜렉션
    import gc
    gc.collect()

    print("LLM 메모리 해제 완료")


# 지원 언어 설정 - Stop words에 한글 문자 추가
SUPPORTED_LANGUAGES = {
    'english': {
        'name': 'English',
        'code': 'en',
        'prompt_name': 'English',
        'stop_words': ["\n\n", "Korean:", "English:", "가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파",
                       "하"]
    },
    'chinese': {
        'name': 'Chinese',
        'code': 'zh',
        'prompt_name': 'Chinese (Simplified)',
        'stop_words': ["\n\n", "Korean:", "Chinese:", "가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파",
                       "하"],
        'length_multiplier': 1.2
    },
    'japanese': {
        'name': 'Japanese',
        'code': 'ja',
        'prompt_name': 'Japanese',
        'stop_words': ["\n\n", "Korean:", "Japanese:", "가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파",
                       "하"]
    }
}


def _create_enhanced_prompt(text: str, target_language: str, length_guide: str, is_free: bool = False) -> str:
    """향상된 프롬프트 생성 - 한글 출력 방지 강화"""

    # 감탄사나 의성어 감지
    simple_expressions = ["네", "예", "아", "오", "어", "음", "응", "아니", "그래", "맞아", "좋아", "안녕"]
    is_simple = any(expr in text for expr in simple_expressions) and len(text.strip()) <= 10

    # 감정 표현 감지
    emotion_words = ["놀랐", "깜짝", "기뻐", "슬퍼", "화나", "무서워", "좋아", "싫어"]
    has_emotion = any(word in text for word in emotion_words)

    if is_simple or has_emotion:
        # 간단한 표현이나 감정 표현은 더 구체적인 가이드 제공
        base_prompt = (
            f"You are a professional translator. Translate the given Korean expression to {target_language}. "
            f"IMPORTANT: Never output Korean characters. Only provide the {target_language} translation. "
            f"If the Korean text expresses surprise, use appropriate surprise expressions in {target_language}. "
            f"If it's a simple response like '네/예', translate to appropriate response words. "
            f"{length_guide}\n\n"
            f"Korean expression: {text}\n"
            f"Translation in {target_language}:"
        )
    else:
        # 일반 문장
        translation_style = "with natural expressions" if is_free else "accurately"
        base_prompt = (
            f"You are a professional translator. Translate the Korean text to {target_language} {translation_style}. "
            f"CRITICAL RULE: Absolutely no Korean characters in your response. "
            f"Only output the {target_language} translation. "
            f"{length_guide}\n\n"
            f"Korean: {text}\n"
            f"{target_language} translation:"
        )

    return base_prompt


def literal_translate(text: str, max_length_ratio: float = 1.0, quality_mode: str = "balanced",
                      target_lang: str = "english") -> str:
    """
    직역: 한글 출력 방지를 강화한 번역
    """
    if target_lang not in SUPPORTED_LANGUAGES:
        target_lang = "english"

    lang_config = SUPPORTED_LANGUAGES[target_lang]
    target_language = lang_config['name']
    stop_words = lang_config['stop_words']

    # 길이 가이드 단순화
    if max_length_ratio < 0.8:
        length_guide = "Keep the translation concise and brief."
    else:
        length_guide = "Make the translation natural and fluent."

    # 향상된 프롬프트 사용
    prompt = _create_enhanced_prompt(text, target_language, length_guide, False)

    llm = _get_llm()

    # GPU 메모리 사용량 확인 (번역 시작 전)
    gpu_memory_before = get_gpu_memory_usage()
    if gpu_memory_before > 0:
        print(f"🔋 GPU 메모리 사용량 (번역 전): {gpu_memory_before} MB")

    # 더 엄격한 매개변수로 첫 번째 시도
    resp = llm(
        prompt,
        max_tokens=128,  # 토큰 수 줄여서 한글 출력 가능성 감소
        temperature=0.1,  # 온도 더 낮춤
        top_p=0.8,
        stop=stop_words,
        repeat_penalty=1.1  # 반복 방지
    )
    result = resp["choices"][0]["text"].strip().strip('"\'')
    cleaned = _cleanup(result)

    # 번역 결과 검증
    if cleaned.strip() and len(cleaned.strip()) > 0 and not _contains_korean(cleaned):
        print(f"[성공] 직역 완료: {text} → {cleaned}")
        return cleaned

    # 재시도 (더 강력한 프롬프트)
    print(f"[재시도] 더 강력한 프롬프트로 재시도: {result[:30]}...")
    _reset_llm_context()

    # 더 강력한 프롬프트
    strong_prompt = (
        f"TASK: Korean to {target_language} translation\n"
        f"RULE: Absolutely NO Korean characters in output\n"
        f"INPUT: {text}\n"
        f"OUTPUT ({target_language} only):"
    )

    resp = llm(
        strong_prompt,
        max_tokens=64,  # 더 짧게
        temperature=0.05,  # 거의 결정적
        top_p=0.7,
        stop=stop_words,
        repeat_penalty=1.2
    )
    result = resp["choices"][0]["text"].strip().strip('"\'')
    cleaned = _cleanup(result)

    if cleaned.strip() and not _contains_korean(cleaned):
        print(f"[재시도 성공] 직역 완료: {cleaned}")
        return cleaned

    # 마지막으로 사전 기반 번역 시도
    print(f"[사전 번역] Gemma-3 실패, 사전 기반 번역 사용: {text}")
    return _enhanced_fallback_translate(text, target_lang)


def free_translate(text: str, max_length_ratio: float = 1.0, quality_mode: str = "balanced",
                   target_lang: str = "english") -> str:
    """
    의역: Gemma-3 모델의 성능을 신뢰하고 단순화
    """
    if target_lang not in SUPPORTED_LANGUAGES:
        target_lang = "english"

    lang_config = SUPPORTED_LANGUAGES[target_lang]
    target_language = lang_config['name']
    stop_words = lang_config['stop_words']

    # 의성어/감탄사만 있으면 직역 사용
    orig_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if orig_lines and all(re.fullmatch(r"[가-힣]+[\.!?…]*", ln) for ln in orig_lines):
        return literal_translate(text, max_length_ratio, quality_mode, target_lang)

    # 길이 가이드 단순화
    if max_length_ratio < 0.8:
        length_guide = "Keep the translation concise and brief."
    else:
        length_guide = "Make the translation natural and fluent."

    # 향상된 프롬프트 사용
    prompt = _create_enhanced_prompt(text, target_language, length_guide, True)

    llm = _get_llm()

    # 첫 번째 시도
    resp = llm(
        prompt,
        max_tokens=256,
        temperature=0.4,  # 의역은 조금 더 creative하게
        top_p=0.9,
        stop=stop_words
    )
    result = resp["choices"][0]["text"].strip().strip('"\'')
    cleaned = _cleanup(result)

    # 결과 검증: 한글이 포함되어 있으면 무조건 실패
    if cleaned.strip() and not _contains_korean(cleaned):
        # 한 줄 대본이면 첫 문장만
        if len(orig_lines) == 1:
            return cleaned.split("\n", 1)[0]
        return cleaned

    # 한글이 포함되어 있거나 결과가 부적절한 경우 재시도
    print(f"[재시도] 의역 첫 번째 결과 부적절 (한글 포함): {result[:50]}...")
    _reset_llm_context()

    resp = llm(
        prompt,
        max_tokens=256,
        temperature=0.5,  # 온도 더 높임
        top_p=0.9,
        stop=stop_words
    )
    result = resp["choices"][0]["text"].strip().strip('"\'')
    cleaned = _cleanup(result)

    if cleaned.strip() and not _contains_korean(cleaned):
        if len(orig_lines) == 1:
            return cleaned.split("\n", 1)[0]
        return cleaned

    # 의역 실패 시 직역으로 대체 (fallback 대신)
    print(f"[직역 대체] 의역 실패 (한글 포함), 직역 사용: {text}")
    return literal_translate(text, max_length_ratio, quality_mode, target_lang)


# 편의 함수들 추가
def translate_to_chinese(text: str, translation_type: str = "literal", max_length_ratio: float = 1.0,
                         quality_mode: str = "balanced") -> str:
    """중국어 번역 편의 함수"""
    if translation_type == "free":
        return free_translate(text, max_length_ratio, quality_mode, "chinese")
    else:
        return literal_translate(text, max_length_ratio, quality_mode, "chinese")


def translate_to_japanese(text: str, translation_type: str = "literal", max_length_ratio: float = 1.0,
                          quality_mode: str = "balanced") -> str:
    """일본어 번역 편의 함수"""
    if translation_type == "free":
        return free_translate(text, max_length_ratio, quality_mode, "japanese")
    else:
        return literal_translate(text, max_length_ratio, quality_mode, "japanese")


def translate_to_english(text: str, translation_type: str = "literal", max_length_ratio: float = 1.0,
                         quality_mode: str = "balanced") -> str:
    """영어 번역 편의 함수 (기존 호환성 유지)"""
    if translation_type == "free":
        return free_translate(text, max_length_ratio, quality_mode, "english")
    else:
        return literal_translate(text, max_length_ratio, quality_mode, "english")
