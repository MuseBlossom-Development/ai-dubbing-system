# gtranslate.py

import os
import random
import re
from llama_cpp import Llama

# 모델 파일 경로
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "gemma", "gemma-3-12b-it-q4_0.gguf")

# 전역 LLM 인스턴스
_llm = None
MAX_ATTEMPTS = 2

def _get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_threads=os.cpu_count(),
            n_gpu_layers=-1,
            seed=42,
            library=os.path.join(
                os.path.dirname(__file__),
                "llama.cpp", "build-gpu", "Release", "libllama.dll"
            )
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

def literal_translate(text: str) -> str:
    """직역: 최대 MAX_ATTEMPTS 회까지 시도"""
    prompt = (
        f"아래는 연기 대본의 한 대사입니다. 같은 텍스트라도 감정이 다르면 각각 번역해야 합니다.\n"
        f"텍스트: {text}\n"
        "감정을 살려 자연스럽게 영어로 번역해주세요. 오직 번역 결과만 출력하세요:"
    )
    llm = _get_llm()
    for _ in range(MAX_ATTEMPTS):
        resp = llm(
            prompt,
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
            # stop 시퀀스 제거 → Llama가 끝까지 생성하도록
        )
        out = resp["choices"][0]["text"].strip()
        if out:
            return _cleanup(out)
    return text  # 실패 시 원본 반환

def free_translate(text: str) -> str:
    """의역: 빈 응답시 literal 번역으로 대체"""
    orig_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # 의성어/감탄사만 있으면 굳이 의역하지 않음
    if orig_lines and all(re.fullmatch(r"[가-힣]+[\.!?…]*", ln) for ln in orig_lines):
        return literal_translate(text)

    prompt = (
        f"아래는 연기 대본의 한 대사입니다. 같은 텍스트라도 감정이 다르면 각각 의역해야 합니다.\n"
        f"텍스트: {text}\n"
        "감정을 살려 다양한 영어 표현으로 의역해주세요. 오직 번역 결과만 출력하세요:"
    )
    llm = _get_llm()
    for _ in range(MAX_ATTEMPTS):
        resp = llm(
            prompt,
            max_tokens=256,
            temperature=0.8,
            top_p=0.9,
        )
        out = resp["choices"][0]["text"].strip()
        if out:
            cleaned = _cleanup(out)
            if len(orig_lines) == 1:
                # 한 줄 대본이면 첫 문장만
                return cleaned.split("\n",1)[0]
            return cleaned
    return literal_translate(text)
