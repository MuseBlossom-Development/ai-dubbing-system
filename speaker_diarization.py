import os
import json
import torch
import torchaudio
from pyannote.audio import Pipeline
from pydub import AudioSegment
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import log_message


class SpeakerDiarization:
    def __init__(self):
        """화자 분리 클래스 초기화"""
        self.pipeline = None
        self.audio_file = None
        self.diarization_result = None
        self.speakers = {}

    def load_pipeline(self, hf_token=None):
        """pyannote 파이프라인 로드"""
        try:
            log_message("화자 분리 모델 로딩 중...")

            # Hugging Face 토큰이 필요한 경우
            if hf_token:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
            else:
                # 토큰 없이 시도
                self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

            # GPU 사용 가능시 GPU로 이동
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                log_message("GPU 사용 중")
            else:
                log_message("CPU 사용 중")

            log_message("화자 분리 모델 로드 완료")
            return True

        except Exception as e:
            log_message(f"모델 로드 실패: {e}")
            log_message("Hugging Face 토큰이 필요할 수 있습니다.")
            return False

    def analyze_audio(self, audio_path):
        """음성 파일 분석 및 화자 분리"""
        try:
            if not self.pipeline:
                log_message("파이프라인이 로드되지 않았습니다.")
                return False

            self.audio_file = audio_path
            log_message(f"음성 파일 분석 중: {audio_path}")

            # 화자 분리 실행
            self.diarization_result = self.pipeline(audio_path)

            # 결과 분석
            speakers_info = self._analyze_speakers()
            log_message(f"발견된 화자 수: {len(speakers_info)}")

            for speaker_id, info in speakers_info.items():
                log_message(f"화자 {speaker_id}: 총 {info['duration']:.1f}초 발화")

            return True

        except Exception as e:
            log_message(f"음성 분석 실패: {e}")
            return False

    def _analyze_speakers(self):
        """화자별 정보 분석"""
        speakers_info = defaultdict(lambda: {'segments': [], 'duration': 0})

        for turn, _, speaker in self.diarization_result.itertracks(yield_label=True):
            speakers_info[speaker]['segments'].append({
                'start': turn.start,
                'end': turn.end,
                'duration': turn.end - turn.start
            })
            speakers_info[speaker]['duration'] += turn.end - turn.start

        # 발화 시간 순으로 정렬
        self.speakers = dict(sorted(speakers_info.items(),
                                    key=lambda x: x[1]['duration'], reverse=True))

        return self.speakers

    def extract_speaker_segments(self, output_dir):
        """화자별 음성 세그먼트 추출"""
        try:
            if not self.diarization_result or not self.audio_file:
                log_message("분석 결과가 없습니다.")
                return False

            # 출력 디렉토리 생성
            os.makedirs(output_dir, exist_ok=True)

            # 원본 오디오 로드
            audio = AudioSegment.from_file(self.audio_file)

            # 화자별 세그먼트 추출
            for speaker_id, info in self.speakers.items():
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

                log_message(f"화자 {speaker_id} 세그먼트 추출 완료: {len(info['segments'])}개")

            return True

        except Exception as e:
            log_message(f"세그먼트 추출 실패: {e}")
            return False

    def generate_timeline_report(self, output_dir):
        """타임라인 리포트 생성"""
        try:
            if not self.diarization_result:
                return False

            # 타임라인 데이터 생성
            timeline = []
            for turn, _, speaker in self.diarization_result.itertracks(yield_label=True):
                timeline.append({
                    'start': round(turn.start, 2),
                    'end': round(turn.end, 2),
                    'duration': round(turn.end - turn.start, 2),
                    'speaker': speaker
                })

            # 시간순 정렬
            timeline.sort(key=lambda x: x['start'])

            # JSON 리포트 저장
            report_path = os.path.join(output_dir, "diarization_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'audio_file': self.audio_file,
                    'total_speakers': len(self.speakers),
                    'speakers_info': self.speakers,
                    'timeline': timeline
                }, f, indent=2, ensure_ascii=False)

            # 텍스트 리포트 생성
            txt_report_path = os.path.join(output_dir, "diarization_report.txt")
            with open(txt_report_path, 'w', encoding='utf-8') as f:
                f.write("=== 화자 분리 리포트 ===\n\n")
                f.write(f"음성 파일: {self.audio_file}\n")
                f.write(f"총 화자 수: {len(self.speakers)}\n\n")

                f.write("=== 화자별 정보 ===\n")
                for speaker_id, info in self.speakers.items():
                    f.write(f"\n화자 {speaker_id}:\n")
                    f.write(f"  총 발화 시간: {info['duration']:.1f}초\n")
                    f.write(f"  발화 횟수: {len(info['segments'])}회\n")

                f.write("\n=== 타임라인 ===\n")
                for item in timeline:
                    f.write(f"{item['start']:6.1f}s - {item['end']:6.1f}s | "
                            f"화자 {item['speaker']} ({item['duration']:.1f}초)\n")

            log_message(f"리포트 생성 완료: {report_path}")
            return True

        except Exception as e:
            log_message(f"리포트 생성 실패: {e}")
            return False

    def create_visualization(self, output_dir):
        """화자 분리 시각화"""
        try:
            if not self.diarization_result:
                return False

            plt.figure(figsize=(15, 8))

            # 색상 맵 생성
            speakers = list(self.speakers.keys())
            colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
            speaker_colors = dict(zip(speakers, colors))

            # 타임라인 그리기
            y_pos = 0
            for turn, _, speaker in self.diarization_result.itertracks(yield_label=True):
                plt.barh(y_pos, turn.end - turn.start, left=turn.start,
                         color=speaker_colors[speaker], alpha=0.7,
                         label=f'화자 {speaker}' if speaker not in plt.gca().get_legend_handles_labels()[1] else "")
                y_pos += 1

            plt.xlabel('시간 (초)')
            plt.ylabel('발화 순서')
            plt.title('화자별 발화 타임라인')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            viz_path = os.path.join(output_dir, "diarization_timeline.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 화자별 발화 시간 파이차트
            plt.figure(figsize=(10, 8))
            speaker_names = [f'화자 {speaker}' for speaker in self.speakers.keys()]
            durations = [info['duration'] for info in self.speakers.values()]

            plt.pie(durations, labels=speaker_names, autopct='%1.1f%%', startangle=90)
            plt.title('화자별 발화 시간 비율')

            pie_path = os.path.join(output_dir, "speaker_duration_ratio.png")
            plt.savefig(pie_path, dpi=300, bbox_inches='tight')
            plt.close()

            log_message(f"시각화 완료: {viz_path}, {pie_path}")
            return True

        except Exception as e:
            log_message(f"시각화 실패: {e}")
            return False


def test_speaker_diarization(audio_path, hf_token=None):
    """화자 분리 테스트 함수"""
    try:
        # 출력 디렉토리 생성
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join(os.getcwd(), 'speaker_analysis', base_name)
        os.makedirs(output_dir, exist_ok=True)

        # 화자 분리 객체 생성
        diarizer = SpeakerDiarization()

        # 파이프라인 로드
        if not diarizer.load_pipeline(hf_token):
            return False

        # 음성 분석
        if not diarizer.analyze_audio(audio_path):
            return False

        # 세그먼트 추출
        if not diarizer.extract_speaker_segments(output_dir):
            return False

        # 리포트 생성
        if not diarizer.generate_timeline_report(output_dir):
            return False

        # 시각화 생성
        if not diarizer.create_visualization(output_dir):
            return False

        log_message(f"✅ 화자 분리 완료: {output_dir}")
        return True

    except Exception as e:
        log_message(f"화자 분리 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    # 테스트 실행
    import sys

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        hf_token = sys.argv[2] if len(sys.argv) > 2 else None
        test_speaker_diarization(audio_file, hf_token)
    else:
        print("사용법: python speaker_diarization.py <audio_file> [hf_token]")
