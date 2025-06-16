import argparse
import os
from nemo.collections.asr.models import ClusteringDiarizer

def parse_args():
    parser = argparse.ArgumentParser(description="NeMo Speaker Diarization Example")
    parser.add_argument(
        "--audio", "-a",
        required=True,
        help="분리할 오디오 파일 경로 (wav)"
    )
    parser.add_argument(
        "--output", "-o",
        default="diarization_result.txt",
        help="결과 txt 저장 파일명"
    )
    return parser.parse_args()

def rttm_to_txt(rttm_path, txt_path):
    with open(rttm_path, "r", encoding="utf-8") as f_rttm, \
         open(txt_path, "w", encoding="utf-8") as f_txt:
        for line in f_rttm:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            # RTTM: SPEAKER <file-id> <chan> <start> <duration> <ortho> <stype> <name> <conf> <slat>
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            f_txt.write(f"[{start:.2f} ~ {start+duration:.2f}초] 화자 {speaker}\n")

def main():
    args = parse_args()

    # 1. diarization config 다운로드 (최초 1회만 필요)
    config_url = "https://huggingface.co/nvidia/speaker-diarization/resolve/main/configs/diar_infer_meeting.yaml"
    config_path = "diar_infer_meeting.yaml"
    if not os.path.exists(config_path):
        import requests
        with open(config_path, "wb") as f:
            f.write(requests.get(config_url).content)

    # 2. 모델 준비 및 추론 실행
    diarizer = ClusteringDiarizer(cfg=config_path)
    diarizer.diarize(audio_paths=[args.audio])

    # 3. RTTM 결과 찾기
    out_dir = diarizer.output_dir if hasattr(diarizer, "output_dir") else "output"
    rttm_path = None
    for file in os.listdir(out_dir):
        if file.endswith(".rttm"):
            rttm_path = os.path.join(out_dir, file)
            break
    if rttm_path is None:
        print("RTTM 결과 파일을 찾을 수 없습니다.")
        return

    # 4. RTTM -> TXT 변환
    rttm_to_txt(rttm_path, args.output)
    print(f"화자 분리 결과가 {args.output}에 저장되었습니다.")

if __name__ == "__main__":
    main()