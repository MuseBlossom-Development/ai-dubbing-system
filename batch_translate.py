# batch_translate.py

import os
from gtranslate import literal_translate, free_translate, SUPPORTED_LANGUAGES


def batch_translate(input_dir: str, output_dir: str, length_ratio: float = 0.8, target_languages: list = None):
    """
    input_dir: .txt 파일들이 들어있는 폴더 경로 (한국어 대본)
    output_dir: 번역 결과를 저장할 폴더 경로  
    length_ratio: 원본 대비 번역 길이 비율 (0.8 = 원본의 80% 길이로 축약)
    target_languages: 번역할 언어 리스트 (기본값: ["english"])
    """
    if target_languages is None:
        target_languages = ["english"]

    # 지원되지 않는 언어 필터링
    target_languages = [lang for lang in target_languages if lang in SUPPORTED_LANGUAGES]
    if not target_languages:
        target_languages = ["english"]

    os.makedirs(output_dir, exist_ok=True)

    # 각 언어별로 디렉토리 생성
    lang_dirs = {}
    for lang in target_languages:
        lang_name = SUPPORTED_LANGUAGES[lang]['name'].lower()
        lang_base_dir = os.path.join(output_dir, lang_name)
        os.makedirs(lang_base_dir, exist_ok=True)

        lit_dir = os.path.join(lang_base_dir, 'literal')
        free_dir = os.path.join(lang_base_dir, 'free')
        os.makedirs(lit_dir, exist_ok=True)
        os.makedirs(free_dir, exist_ok=True)

        lang_dirs[lang] = {
            'literal': lit_dir,
            'free': free_dir
        }

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.txt'):
            continue

        src_path = os.path.join(input_dir, fname)
        with open(src_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            print(f"[SKIP] {fname}: 파일이 비어 있습니다.")
            continue

        print(f"Processing file: {fname} (길이 비율: {length_ratio:.1f})")

        # 각 언어별로 번역 처리
        for target_lang in target_languages:
            lang_name = SUPPORTED_LANGUAGES[target_lang]['name']
            print(f"  → Translating to {lang_name}...")

            # 1) 직역 (길이 제한 적용)
            lit_out = literal_translate(content, max_length_ratio=length_ratio, target_lang=target_lang)
            out_lit = os.path.join(lang_dirs[target_lang]['literal'], fname)
            with open(out_lit, 'w', encoding='utf-8') as f:
                f.write(lit_out)

            # 2) 의역 (길이 제한 적용)  
            free_out = free_translate(content, max_length_ratio=length_ratio, target_lang=target_lang)
            out_free = os.path.join(lang_dirs[target_lang]['free'], fname)
            with open(out_free, 'w', encoding='utf-8') as f:
                f.write(free_out)

            # 3) 로그
            print(f"    [OK] {lang_name} → literal: {out_lit}, free: {out_free}")
            print(f"    [ORIGINAL] {content} ({len(content)} chars)")
            print(f"    [LITERAL]  {lit_out} ({len(lit_out)} chars)")
            print(f"    [FREE]     {free_out} ({len(free_out)} chars)")
            if lit_out == free_out:
                print(f"    [COMPARE] {fname} ({lang_name}): 직역과 의역 동일")
            else:
                print(f"    [COMPARE] {fname} ({lang_name}): 직역 vs 의역 다름")
            print("    " + "-" * 30)

        print("-" * 40)

    # ——— 번역 완료 후 즉시 LLM 메모리 정리 ———
    print("[메모리 정리] Gemma3 모델을 메모리에서 해제합니다...")
    cleanup_llm_memory()
    print("[메모리 정리] 완료!")


def batch_translate_multi_lang(input_dir: str, output_dir: str, length_ratio: float = 0.8):
    """모든 지원 언어로 번역하는 편의 함수"""
    all_languages = list(SUPPORTED_LANGUAGES.keys())
    batch_translate(input_dir, output_dir, length_ratio, all_languages)


def batch_translate_english(input_dir: str, output_dir: str, length_ratio: float = 0.8):
    """영어만 번역하는 편의 함수 (기존 호환성)"""
    batch_translate(input_dir, output_dir, length_ratio, ["english"])


def batch_translate_chinese(input_dir: str, output_dir: str, length_ratio: float = 0.8):
    """중국어만 번역하는 편의 함수"""
    batch_translate(input_dir, output_dir, length_ratio, ["chinese"])


def batch_translate_japanese(input_dir: str, output_dir: str, length_ratio: float = 0.8):
    """일본어만 번역하는 편의 함수"""
    batch_translate(input_dir, output_dir, length_ratio, ["japanese"])


def cleanup_llm_memory():
    """LLM 메모리 정리 함수"""
    import gc

    try:
        # gtranslate 모듈에서 _llm 가져오기
        from gtranslate import _llm

        # 1) _llm 인스턴스 닫기/삭제
        if _llm is not None:
            try:
                # llama_cpp의 Llama 객체 해제
                if hasattr(_llm, 'close'):
                    _llm.close()
                elif hasattr(_llm, '__del__'):
                    _llm.__del__()
            except Exception as e:
                print(f"LLM 객체 닫기 중 오류 (무시): {e}")

            # 전역 변수 초기화
            import gtranslate
            gtranslate._llm = None

        # 2) 가비지 콜렉션 (여러 번 실행으로 확실하게)
        for _ in range(3):
            collected = gc.collect()
            print(f"   가비지 콜렉션: {collected}개 객체 해제")

        # 3) CUDA 캐시 비우기 (만약 CUDA를 사용했다면)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   CUDA 캐시 정리 완료")
        except ImportError:
            pass  # torch가 없으면 무시

    except Exception as e:
        print(f"메모리 정리 중 오류 (무시): {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python batch_translate.py <input_dir> <output_dir> [length_ratio] [languages]")
        print("  languages: comma-separated list (english,chinese,japanese) or 'all'")
        print("Examples:")
        print("  python batch_translate.py input output 0.8 english")
        print("  python batch_translate.py input output 0.8 english,chinese")
        print("  python batch_translate.py input output 0.8 all")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    length_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

    # 언어 파라미터 처리
    if len(sys.argv) > 4:
        lang_param = sys.argv[4].lower()
        if lang_param == "all":
            target_languages = list(SUPPORTED_LANGUAGES.keys())
        else:
            target_languages = [lang.strip() for lang in lang_param.split(',')]
    else:
        target_languages = ["english"]  # 기본값

    print(f"Target languages: {', '.join(target_languages)}")
    batch_translate(input_dir, output_dir, length_ratio, target_languages)

    # ——— 여기서 LLM 정리 ———
    import gc, torch
    from gtranslate import _llm

    # 1) _llm 인스턴스 닫기/삭제
    if _llm is not None:
        try:
            _llm.close()
        except Exception:
            pass
        del globals()["_llm"]

    # 2) 가비지 콜렉션
    gc.collect()

    # 3) CUDA 캐시 비우기
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
