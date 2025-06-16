# test_translate.py
from translate import literal_translate, free_translate

def run_tests():
    samples = [
        "오늘 날씨가 정말 좋네요.",
        "저는 내일 친구를 만나러 갑니다.",
        "이 프로젝트는 내일까지 끝내야 합니다.",
        "한국어와 영어의 뉘앙스 차이를 이해하고 싶어요."
    ]

    for idx, text in enumerate(samples, 1):
        print(f"\n샘플 #{idx}: {text}\n" + "-"*40)
        lit = literal_translate(text)
        free = free_translate(text)
        print("▶ 직역 (Literal):", lit)
        print("▶ 의역 (Free):  ", free)

if __name__ == "__main__":
    run_tests()