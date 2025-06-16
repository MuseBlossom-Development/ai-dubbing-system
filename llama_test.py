# file: run_gemma3.py
from llama_cpp import Llama

def main():
    # 1) 모델 경로 설정
    model_path = "/models/gemma-3-27b-gguf/gemma-3-27b-it-qat-q4_0-gguf"
    
    # 2) Llama 객체 생성
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,        # 최대 컨텍스트 길이
        seed=42,           # 재현을 위한 시드
        n_threads=8        # CPU 스레드 수
        # gpu_layers=50    # GPU 가속 레이어 수 설정 (llama-cpp-python v2+)
    )
    
    # 3) 프롬프트 정의
    prompt = (
        "안녕하세요, Gemma 3에게 질문드립니다.\n"
        "파이썬 예제 코드를 보여주세요."
    )
    
    # 4) 텍스트 생성
    resp = llm(
        prompt,
        max_tokens=128,
        temperature=0.7,
        top_p=0.9,
        stream=False      # 토큰 단위 스트리밍이 필요하면 True로 설정
    )
    
    # 5) 결과 출력
    print("=== Generated ===")
    print(resp["choices"][0]["text"])

if __name__ == "__main__":
    main()