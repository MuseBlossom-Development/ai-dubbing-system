import os
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch

def get_hf_token():
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except ImportError:
        pass
    raise RuntimeError("Hugging Face 토큰을 찾을 수 없습니다.")

def get_folder_name(file_path):
    base = os.path.basename(file_path)
    return os.path.splitext(base)[0]

def diarization_and_timeline_mask(input_path):
    hf_token = get_hf_token()
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    pipeline.to(torch.device("cuda"))  # pyannote 3.x는 자동 GPU/CPU 감지

    audio = AudioSegment.from_file(input_path)
    duration = len(audio) / 1000.0

    # diarization 수행 (화자 수 자동 추정)
    diarization = pipeline(input_path)

    speakers = sorted(set(label for _, _, label in diarization.itertracks(yield_label=True)))
    print("Detected speakers:", speakers)

    # 전체 타임라인 마스킹
    total_length = len(audio)
    speaker_masked = {speaker: AudioSegment.silent(duration=total_length, frame_rate=audio.frame_rate) for speaker in speakers}
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        before = speaker_masked[speaker][:start_ms]
        middle = audio[start_ms:end_ms]
        after = speaker_masked[speaker][end_ms:]
        speaker_masked[speaker] = before + middle + after

    folder_name = get_folder_name(input_path)
    output_dir = os.path.join("stem_output", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    for speaker, seg in speaker_masked.items():
        output_file = os.path.join(output_dir, f"{speaker}_full.wav")
        seg.export(output_file, format="wav")
        print(f"Saved: {output_file} ({len(seg)//1000}초)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="pyannote.audio 3.x용 diarization & 타임라인 마스킹")
    parser.add_argument("audio", help="입력 오디오(wav/mp3)")
    args = parser.parse_args()
    diarization_and_timeline_mask(args.audio)