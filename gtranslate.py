# gtranslate.py

import os
import platform
import random
import re
import logging
from llama_cpp import Llama

# Numba 디버그 로그 억제
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.core').setLevel(logging.WARNING)
logging.getLogger('numba.typed').setLevel(logging.WARNING)

# 모델 파일 경로
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "gemma", "gemma-3-12b-it-q4_0.gguf")

# 전역 LLM 인스턴스
_llm = None
MAX_ATTEMPTS = 5  # 재시도 횟수 증가


def _get_library_path():
    """OS에 따른 라이브러리 경로 반환"""
    system = platform.system()
    base_path = os.path.dirname(__file__)

    if system == "Windows":
        # Windows: build-gpu/Release/libllama.dll
        return os.path.join(base_path, "llama.cpp", "build-gpu", "Release", "libllama.dll")
    elif system == "Darwin":  # macOS
        # macOS: build/bin/libllama.dylib
        return os.path.join(base_path, "llama.cpp", "build", "bin", "libllama.dylib")
    else:  # Linux
        # Linux: build/bin/libllama.so
        return os.path.join(base_path, "llama.cpp", "build", "bin", "libllama.so")


def _get_llm():
    global _llm
    if _llm is None:
        library_path = _get_library_path()

        # 라이브러리 파일 존재 확인
        if not os.path.exists(library_path):
            print(f"Warning: Library not found at {library_path}")
            library_path = None  # None으로 설정하면 시스템 기본 라이브러리 사용

        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=os.cpu_count(),
            n_gpu_layers=-1,
            seed=None,  # 매번 다른 시드 사용하여 컨텍스트 초기화
            library=library_path,
            verbose=False  # 불필요한 출력 억제
        )
    return _llm


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


def _fallback_translate(text: str, target_lang: str) -> str:
    """간단한 fallback 번역 함수"""
    # 간단한 영어 대응 단어 사전 (기본 fallback용)
    simple_dict = {
        "맞습니다": "Yes", "네": "Yes", "안녕하세요": "Hello", "감사합니다": "Thank you",
        "어르신": "elders", "저희": "we", "상품": "product", "개발": "developed",
        "쉽게": "easily", "이해": "understand", "참여": "participate", "특별히": "specially",
        "있어요": "have", "그룹": "Group", "디지털": "digital", "투자": "investment",
        "전문": "specialized", "계열사": "affiliate", "당연하죠": "of course",
        "평생": "lifetime", "열심히": "diligently", "모으신": "saved", "분들": "people",
        "너무": "so", "속상하시죠": "upset"
    }

    # 기본 번역 시도 - 단어 기반 매칭
    try:
        words = text.split()
        translated_words = []

        for word in words:
            # 특수문자 제거한 clean_word
            clean_word = re.sub(r'[^\w가-힣]', '', word)

            # 사전에서 찾기
            translated = None
            for korean, english in simple_dict.items():
                if korean in clean_word:
                    if target_lang == "english":
                        translated = english
                    elif target_lang == "chinese":
                        # 간단한 중국어 매핑
                        cn_dict = {"Yes": "是的", "Hello": "你好", "Thank you": "谢谢",
                                   "we": "我们", "have": "有", "Group": "集团"}
                        translated = cn_dict.get(english, english)
                    elif target_lang == "japanese":
                        # 간단한 일본어 매핑
                        jp_dict = {"Yes": "はい", "Hello": "こんにちは", "Thank you": "ありがとう",
                                   "we": "私たち", "have": "あります", "Group": "グループ"}
                        translated = jp_dict.get(english, english)
                    break

            if translated:
                translated_words.append(translated)

        if translated_words:
            return " ".join(translated_words)

    except Exception as e:
        print(f"단순 번역 오류: {e}")

    # 완전 실패 시 기본 응답 반환 (언어별)
    if target_lang == "english":
        return "Translation unavailable"
    elif target_lang == "chinese":
        return "翻译不可用"
    elif target_lang == "japanese":
        return "翻訳できません"
    else:
        return "Translation error"


# 지원 언어 설정
SUPPORTED_LANGUAGES = {
    'english': {
        'name': 'English',
        'code': 'en',
        'prompt_name': 'English',
        'stop_words': ["Korean:", "English:", "한국어:", "영어:"]  # "\n" 제거하여 문장 잘림 방지
    },
    'chinese': {
        'name': 'Chinese',
        'code': 'zh',
        'prompt_name': 'Chinese (Simplified)',
        'stop_words': ["Korean:", "Chinese:", "한국어:", "中文:", "中国语:"],  # 중국어 정지어 개선
        'length_multiplier': 1.2  # 중국어는 더 긴 번역을 위한 배수
    },
    'japanese': {
        'name': 'Japanese',
        'code': 'ja',
        'prompt_name': 'Japanese',
        'stop_words': ["Korean:", "Japanese:", "한국어:", "日本語:", "일본어:"]  # "\n" 제거
    }
}


def literal_translate(text: str, max_length_ratio: float = 1.0, quality_mode: str = "balanced",
                      target_lang: str = "english") -> str:
    """
    직역: 최대 MAX_ATTEMPTS 회까지 시도
    
    Args:
        text: 번역할 한국어 텍스트
        max_length_ratio: 원본 대비 최대 길이 비율 (1.0 = 원본과 같은 길이)
        quality_mode: "concise" (간결함 우선), "balanced" (균형), "accurate" (정확성 우선)
        target_lang: 번역 대상 언어 ("english", "chinese", "japanese")
    """
    if target_lang not in SUPPORTED_LANGUAGES:
        target_lang = "english"

    lang_config = SUPPORTED_LANGUAGES[target_lang]
    target_language = lang_config['prompt_name']
    stop_words = lang_config['stop_words']

    # 길이 가이드 문구 생성 (언어별 특화)
    if max_length_ratio < 1.0:
        if max_length_ratio <= 0.7:
            if target_lang == "english":
                length_guide = "Keep the translation very concise and brief - use fewer words than the original Korean text. Prefer short, direct expressions."
            else:
                length_guide = "Keep the translation very concise and brief - use fewer words than the original Korean text."
        else:
            if target_lang == "english":
                length_guide = "Keep the translation similar in length to the original Korean text - avoid making it much longer. Use concise English expressions."
            else:
                length_guide = "Keep the translation similar in length to the original Korean text - avoid making it much longer."
    else:
        if target_lang == "english":
            length_guide = "Keep the translation concise and natural - avoid making it unnecessarily long. Use efficient English expressions."
        else:
            length_guide = "Keep the translation concise and natural - avoid making it unnecessarily long."

    # 중국어 번역 길이 조정
    if target_lang == "chinese":
        # 중국어는 한국어 대비 약 0.3배 길이로, 원본 길이 대비 1.2배로 조정
        length_guide += " Chinese translation should be about 1.2 times the length of the Korean text."

    # 품질 모드에 따른 프롬프트 조정
    if quality_mode == "concise":
        if target_lang == "chinese":
            prompt = (
                f"Translate the following Korean text to natural {target_language}. "
                f"{length_guide} "
                f"Use detailed and polite Chinese expressions that match the Korean speaking style. "
                f"Include appropriate honorifics and explanatory phrases. Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        else:
            prompt = (
                f"Translate the following Korean text to {target_language}. "
                f"{length_guide} "
                f"Use simple and clear expressions. Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        max_tokens = 512
        temperature = 0.2
    elif quality_mode == "accurate":
        if target_lang == "chinese":
            prompt = (
                f"Translate the following Korean text to natural {target_language}. "
                f"{length_guide} "
                f"Preserve all nuances, honorifics, and speaking style in Chinese. "
                f"Use detailed expressions that maintain the original Korean tone and politeness level. "
                f"Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        else:
            prompt = (
                f"Translate the following Korean text to {target_language}. "
                f"{length_guide} "
                f"Preserve all nuances and meaning accurately. Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        max_tokens = 512
        temperature = 0.3
    else:  # balanced
        if target_lang == "chinese":
            prompt = (
                f"Translate the following Korean text to natural {target_language}. "
                f"{length_guide} "
                f"Use natural and detailed Chinese expressions that capture the Korean speaking style. "
                f"Include appropriate honorifics and maintain the conversational tone. "
                f"Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        else:
            prompt = (
                f"Translate the following Korean text to {target_language}. "
                f"{length_guide} "
                f"Make it natural and fluent. Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        max_tokens = 512
        temperature = 0.3

    llm = _get_llm()
    for attempt in range(MAX_ATTEMPTS + 2):
        resp = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
            stop=stop_words
        )
        out = resp["choices"][0]["text"].strip()

        if out:
            # 따옴표 제거
            cleaned_out = out.strip('"\'')

            # 한글 포함 여부만 간단히 확인
            if _contains_korean(cleaned_out):
                print(f"[경고] 번역 결과에 한글 포함됨 (시도 {attempt + 1}): {cleaned_out}")
                continue

            cleaned = _cleanup(cleaned_out)
            if cleaned.strip():
                return cleaned

    # 모든 시도 실패 시 단순 번역 시도
    print(f"[경고] 번역 실패, 단순 번역 시도: {text}")
    return _fallback_translate(text, target_lang)


def free_translate(text: str, max_length_ratio: float = 1.0, quality_mode: str = "balanced",
                   target_lang: str = "english") -> str:
    """
    의역: 빈 응답시 literal 번역으로 대체
    
    Args:
        text: 번역할 한국어 텍스트  
        max_length_ratio: 원본 대비 최대 길이 비율 (1.0 = 원본과 같은 길이)
        quality_mode: "concise" (간결함 우선), "balanced" (균형), "accurate" (정확성 우선)
        target_lang: 번역 대상 언어 ("english", "chinese", "japanese")
    """
    if target_lang not in SUPPORTED_LANGUAGES:
        target_lang = "english"

    lang_config = SUPPORTED_LANGUAGES[target_lang]
    target_language = lang_config['prompt_name']
    stop_words = lang_config['stop_words']

    orig_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 의성어/감탄사만 있으면 굳이 의역하지 않음
    if orig_lines and all(re.fullmatch(r"[가-힣]+[\.!?…]*", ln) for ln in orig_lines):
        return literal_translate(text, max_length_ratio, quality_mode, target_lang)

    # 길이 가이드 문구 생성 (언어별 특화)
    if max_length_ratio < 1.0:
        if max_length_ratio <= 0.7:
            if target_lang == "english":
                length_guide = "Keep the translation very concise and brief - use fewer words than the original Korean text. Prefer short, direct expressions."
            else:
                length_guide = "Keep the translation very concise and brief - use fewer words than the original Korean text."
        else:
            if target_lang == "english":
                length_guide = "Keep the translation similar in length to the original Korean text - avoid making it much longer. Use concise English expressions."
            else:
                length_guide = "Keep the translation similar in length to the original Korean text - avoid making it much longer."
    else:
        if target_lang == "english":
            length_guide = "Keep the translation concise and natural - avoid making it unnecessarily long. Use efficient English expressions."
        else:
            length_guide = "Keep the translation concise and natural - avoid making it unnecessarily long."

    # 중국어 번역 길이 조정
    if target_lang == "chinese":
        # 중국어는 한국어 대비 약 0.3배 길이로, 원본 길이 대비 1.2배로 조정
        length_guide += " Chinese translation should be about 1.2 times the length of the Korean text."

    # 품질 모드에 따른 프롬프트 조정
    if quality_mode == "concise":
        if target_lang == "chinese":
            prompt = (
                f"Translate the following Korean text to natural {target_language}. "
                f"{length_guide} "
                f"Use detailed and polite Chinese expressions that match the Korean speaking style. "
                f"Include appropriate honorifics and explanatory phrases. Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        else:
            prompt = (
                f"Translate the following Korean text to {target_language}. "
                f"{length_guide} "
                f"Use brief and natural expressions. Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        max_tokens = 512
        temperature = 0.4
    elif quality_mode == "accurate":
        if target_lang == "chinese":
            prompt = (
                f"Translate the following Korean text to natural {target_language}. "
                f"{length_guide} "
                f"Preserve all nuances, honorifics, and speaking style in Chinese. "
                f"Use detailed expressions that maintain the original Korean tone and politeness level. "
                f"Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        else:
            prompt = (
                f"Translate the following Korean text to natural {target_language}. "
                f"{length_guide} "
                f"Preserve meaning with natural expressions. Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        max_tokens = 512
        temperature = 0.5
    else:  # balanced
        if target_lang == "chinese":
            prompt = (
                f"Translate the following Korean text to natural {target_language}. "
                f"{length_guide} "
                f"Use natural and detailed Chinese expressions that capture the Korean speaking style. "
                f"Include appropriate honorifics and maintain the conversational tone. "
                f"Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        else:
            prompt = (
                f"Translate the following Korean text to natural {target_language}. "
                f"{length_guide} "
                f"Use natural and fluent expressions. Provide ONLY the translation.\n\n"
                f"Korean: {text}\n"
                f"{target_language}:"
            )
        max_tokens = 512
        temperature = 0.5

    llm = _get_llm()
    for attempt in range(MAX_ATTEMPTS + 2):
        resp = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
            stop=stop_words
        )
        out = resp["choices"][0]["text"].strip()

        if out:
            # 따옴표 제거
            cleaned_out = out.strip('"\'')

            # 한글 포함 여부만 간단히 확인
            if _contains_korean(cleaned_out):
                print(f"[경고] 의역 결과에 한글 포함됨 (시도 {attempt + 1}): {cleaned_out}")
                continue

            cleaned = _cleanup(cleaned_out)
            if cleaned.strip() and not _contains_korean(cleaned):
                if len(orig_lines) == 1:
                    # 한 줄 대본이면 첫 문장만
                    return cleaned.split("\n", 1)[0]
                return cleaned

    # 의역 실패 시 직역으로 대체
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
