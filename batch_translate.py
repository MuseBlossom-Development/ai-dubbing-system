# batch_translate.py

import os
from gtranslate import literal_translate, free_translate

def batch_translate(input_dir: str, output_dir: str):
    """
    input_dir: .txt 파일들이 들어있는 폴더 경로 (한국어 대본)
    output_dir: 번역 결과를 저장할 폴더 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    lit_dir = os.path.join(output_dir, 'literal')
    free_dir = os.path.join(output_dir, 'free')
    os.makedirs(lit_dir, exist_ok=True)
    os.makedirs(free_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.txt'):
            continue

        src_path = os.path.join(input_dir, fname)
        with open(src_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            print(f"[SKIP] {fname}: 파일이 비어 있습니다.")
            continue

        print(f"Processing file: {fname}")

        # 1) 직역
        lit_out = literal_translate(content)
        out_lit = os.path.join(lit_dir, fname)
        with open(out_lit, 'w', encoding='utf-8') as f:
            f.write(lit_out)

        # 2) 의역
        free_out = free_translate(content)
        out_free = os.path.join(free_dir, fname)
        with open(out_free, 'w', encoding='utf-8') as f:
            f.write(free_out)

        # 3) 로그
        print(f"[OK] {fname} → literal: {out_lit}, free: {out_free}")
        print(f"  [ORIGINAL] {content}")
        if lit_out == free_out:
            print(f"  [COMPARE] {fname}: 직역과 의역 동일")
        else:
            print(f"  [COMPARE] {fname}: 직역 vs 의역 다름")
            print(f"    [LITERAL] {lit_out}")
            print(f"    [FREE]    {free_out}")
        print("-" * 40)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python batch_translate.py <input_dir> <output_dir>")
    else:
        batch_translate(sys.argv[1], sys.argv[2])
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
