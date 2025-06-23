#!/usr/bin/env python3
"""
화자 분리 테스트 스크립트
사용법: python test_speaker_diarization.py <audio_file>
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import argparse
from pathlib import Path

try:
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    from pydub import AudioSegment
    import numpy as np
    from collections import defaultdict
    import matplotlib.pyplot as plt

    print("✅ 필요한 라이브러리 로드 완료")
except ImportError as e:
    print(f"❌ 라이브러리 누락: {e}")
    print("다음 명령어로 설치하세요:")
    print("pip install pyannote.audio torch torchaudio pydub matplotlib numpy")
    sys.exit(1)


def analyze_speakers(audio_path, output_dir=None, hf_token=None):
    """화자 분리 분석 실행"""

    if output_dir is None:
        output_dir = f"speaker_analysis_{Path(audio_path).stem}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"🎵 음성 파일: {audio_path}")
    print(f"📁 출력 디렉토리: {output_dir}")

    try:
        # 1. 파이프라인 로드
        print("\n🔄 화자 분리 모델 로딩 중...")

        if hf_token:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
        else:
            # 토큰 없이 시도 (공개 모델인 경우)
            try:
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            except Exception as e:
                print(f"❌ 모델 로드 실패: {e}")
                print("💡 Hugging Face 토큰이 필요할 수 있습니다.")
                print("   https://huggingface.co/pyannote/speaker-diarization-3.1 에서 승인 받으세요")
                return False

        # GPU 사용 가능 시 GPU로 이동
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            print("🚀 GPU 사용 중")
        else:
            print("🐌 CPU 사용 중 (시간이 오래 걸릴 수 있습니다)")

        print("✅ 모델 로드 완료")

        # 2. 화자 분리 실행
        print("\n🔍 음성 분석 중...")
        diarization = pipeline(audio_path)

        # 3. 결과 분석
        speakers_info = defaultdict(lambda: {'segments': [], 'duration': 0})
        timeline = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers_info[speaker]['segments'].append({
                'start': turn.start,
                'end': turn.end,
                'duration': turn.end - turn.start
            })
            speakers_info[speaker]['duration'] += turn.end - turn.start

            timeline.append({
                'start': round(turn.start, 2),
                'end': round(turn.end, 2),
                'duration': round(turn.end - turn.start, 2),
                'speaker': speaker
            })

        # 발화 시간 순으로 정렬
        speakers_info = dict(sorted(speakers_info.items(),
                                    key=lambda x: x[1]['duration'], reverse=True))

        # 시간순 정렬
        timeline.sort(key=lambda x: x['start'])

        print(f"\n📊 분석 결과:")
        print(f"   총 화자 수: {len(speakers_info)}명")

        for speaker_id, info in speakers_info.items():
            print(f"   화자 {speaker_id}: {info['duration']:.1f}초 발화 ({len(info['segments'])}회)")

        # 4. 음성 세그먼트 추출
        print(f"\n✂️  화자별 음성 추출 중...")
        audio = AudioSegment.from_file(audio_path)

        for speaker_id, info in speakers_info.items():
            speaker_dir = os.path.join(output_dir, f"speaker_{speaker_id}")
            os.makedirs(speaker_dir, exist_ok=True)

            # 개별 세그먼트 추출
            for i, segment in enumerate(info['segments']):
                start_ms = int(segment['start'] * 1000)
                end_ms = int(segment['end'] * 1000)

                segment_audio = audio[start_ms:end_ms]
                segment_path = os.path.join(speaker_dir, f"segment_{i + 1:03d}.wav")
                segment_audio.export(segment_path, format="wav")

            # 화자별 전체 음성 병합
            merged_audio = AudioSegment.empty()
            for segment in info['segments']:
                start_ms = int(segment['start'] * 1000)
                end_ms = int(segment['end'] * 1000)
                merged_audio += audio[start_ms:end_ms]

            merged_path = os.path.join(speaker_dir, f"speaker_{speaker_id}_merged.wav")
            merged_audio.export(merged_path, format="wav")

            print(f"   화자 {speaker_id}: {len(info['segments'])}개 세그먼트 추출")

        # 5. 리포트 생성
        print(f"\n📝 리포트 생성 중...")

        # JSON 리포트
        report_data = {
            'audio_file': audio_path,
            'total_speakers': len(speakers_info),
            'speakers_info': {k: {'duration': v['duration'], 'segments_count': len(v['segments'])}
                              for k, v in speakers_info.items()},
            'timeline': timeline
        }

        json_path = os.path.join(output_dir, "analysis_report.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # 텍스트 리포트
        txt_path = os.path.join(output_dir, "analysis_report.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=== 화자 분리 분석 리포트 ===\n\n")
            f.write(f"음성 파일: {audio_path}\n")
            f.write(f"총 화자 수: {len(speakers_info)}명\n\n")

            f.write("=== 화자별 정보 ===\n")
            for speaker_id, info in speakers_info.items():
                f.write(f"\n화자 {speaker_id}:\n")
                f.write(f"  총 발화 시간: {info['duration']:.1f}초\n")
                f.write(f"  발화 횟수: {len(info['segments'])}회\n")
                f.write(f"  평균 발화 길이: {info['duration'] / len(info['segments']):.1f}초\n")

            f.write(f"\n=== 타임라인 ({len(timeline)}개 구간) ===\n")
            for item in timeline:
                f.write(f"{item['start']:7.1f}s - {item['end']:7.1f}s | "
                        f"화자 {item['speaker']} ({item['duration']:.1f}초)\n")

        # 6. 시각화 생성
        print(f"📈 시각화 생성 중...")

        try:
            # 타임라인 시각화
            plt.figure(figsize=(15, 8))
            speakers = list(speakers_info.keys())
            colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
            speaker_colors = dict(zip(speakers, colors))

            y_pos = 0
            legend_added = set()

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                label = f'화자 {speaker}' if speaker not in legend_added else ""
                if speaker not in legend_added:
                    legend_added.add(speaker)

                plt.barh(y_pos, turn.end - turn.start, left=turn.start,
                         color=speaker_colors[speaker], alpha=0.7, label=label)
                y_pos += 1

            plt.xlabel('시간 (초)')
            plt.ylabel('발화 순서')
            plt.title('화자별 발화 타임라인')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            timeline_path = os.path.join(output_dir, "timeline.png")
            plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 발화 시간 비율 파이차트
            plt.figure(figsize=(10, 8))
            speaker_names = [f'화자 {speaker}' for speaker in speakers_info.keys()]
            durations = [info['duration'] for info in speakers_info.values()]

            plt.pie(durations, labels=speaker_names, autopct='%1.1f%%', startangle=90)
            plt.title('화자별 발화 시간 비율')

            pie_path = os.path.join(output_dir, "speaker_ratio.png")
            plt.savefig(pie_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"   시각화 완료: timeline.png, speaker_ratio.png")

        except Exception as e:
            print(f"⚠️  시각화 생성 실패: {e}")

        # 7. 완료 메시지
        print(f"\n🎉 화자 분리 완료!")
        print(f"📁 결과 폴더: {output_dir}")
        print(f"📊 총 {len(speakers_info)}명의 화자 발견")
        print(f"📝 리포트: analysis_report.txt, analysis_report.json")

        return True

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='화자 분리 테스트')
    parser.add_argument('audio_file', help='분석할 음성/영상 파일')
    parser.add_argument('--output', '-o', help='출력 디렉토리 (기본값: speaker_analysis_<filename>)')
    parser.add_argument('--token', help='Hugging Face 토큰 (필요한 경우)')

    if len(sys.argv) == 1:
        # 인수가 없으면 도움말 표시
        parser.print_help()
        print(f"\n예시:")
        print(f"  python {sys.argv[0]} audio.wav")
        print(f"  python {sys.argv[0]} video.mp4 --output my_analysis")
        print(f"  python {sys.argv[0]} audio.wav --token your_hf_token")
        return

    args = parser.parse_args()

    # 파일 존재 확인
    if not os.path.exists(args.audio_file):
        print(f"❌ 파일을 찾을 수 없습니다: {args.audio_file}")
        return

    # 분석 실행
    success = analyze_speakers(
        audio_path=args.audio_file,
        output_dir=args.output,
        hf_token=args.token
    )

    if success:
        print(f"\n✅ 분석 완료!")
    else:
        print(f"\n❌ 분석 실패")


if __name__ == "__main__":
    main()
