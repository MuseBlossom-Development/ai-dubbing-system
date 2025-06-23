import os
import numpy as np
from utils import log_message


def analyze_speakers_simple(segments, audio_path):
    """
    간단한 화자 분석 (librosa 기반)
    
    Args:
        segments: [(start_ms, end_ms), ...] 세그먼트 리스트
        audio_path: 오디오 파일 경로
        
    Returns:
        speaker_labels: [0, 0, 1, 1, 0, ...] 화자 레이블 (같은 숫자 = 같은 화자)
    """
    try:
        import librosa
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity

        log_message("🎤 화자 분석 시작...")

        embeddings = []
        valid_segments = []

        # 각 세그먼트에서 음성 특징 추출
        for i, (start_ms, end_ms) in enumerate(segments):
            try:
                # 세그먼트가 너무 짧으면 건너뛰기
                if end_ms - start_ms < 500:  # 0.5초 미만
                    embeddings.append(None)
                    continue

                # 오디오 로드
                audio, sr = librosa.load(
                    audio_path,
                    offset=start_ms / 1000,
                    duration=(end_ms - start_ms) / 1000,
                    sr=16000  # 16kHz로 통일
                )

                if len(audio) < 1000:  # 너무 짧은 오디오
                    embeddings.append(None)
                    continue

                # 화자 특징 추출 (MFCC + Spectral features)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

                # 특징들을 하나로 합치기
                features = np.concatenate([
                    np.mean(mfcc, axis=1),
                    np.mean(spectral_centroid),
                    np.mean(spectral_rolloff),
                    np.mean(zero_crossing_rate)
                ])

                embeddings.append(features)
                valid_segments.append(i)

            except Exception as e:
                log_message(f"세그먼트 {i} 분석 실패: {e}")
                embeddings.append(None)

        # 유효한 임베딩만 필터링
        valid_embeddings = [emb for emb in embeddings if emb is not None]

        if len(valid_embeddings) < 2:
            log_message("⚠️ 화자 분석을 위한 유효한 세그먼트가 부족합니다. 모든 세그먼트를 같은 화자로 처리합니다.")
            return [0] * len(segments)

        # 클러스터링으로 화자 구분
        # 코사인 유사도 기반 계층 클러스터링
        similarity_matrix = cosine_similarity(valid_embeddings)
        distance_matrix = 1 - similarity_matrix

        # 자동으로 화자 수 결정 (거리 임계값 기반)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.3,  # 화자 구분 민감도 (낮을수록 더 많은 화자)
            metric='precomputed',
            linkage='average'
        )

        cluster_labels = clustering.fit_predict(distance_matrix)

        # 결과를 원본 세그먼트 순서에 맞춰 정리
        speaker_labels = []
        valid_idx = 0

        for i, embedding in enumerate(embeddings):
            if embedding is not None:
                speaker_labels.append(cluster_labels[valid_idx])
                valid_idx += 1
            else:
                # 유효하지 않은 세그먼트는 이전 화자와 같다고 가정
                if speaker_labels:
                    speaker_labels.append(speaker_labels[-1])
                else:
                    speaker_labels.append(0)

        num_speakers = len(set(speaker_labels))
        log_message(f"✅ 화자 분석 완료: {num_speakers}명의 화자 감지")

        # 화자별 세그먼트 수 출력
        for speaker_id in set(speaker_labels):
            count = speaker_labels.count(speaker_id)
            log_message(f"   화자 {speaker_id}: {count}개 세그먼트")

        return speaker_labels

    except ImportError:
        log_message("❌ librosa 또는 sklearn이 설치되지 않았습니다.")
        log_message("pip install librosa scikit-learn 으로 설치해주세요.")
        return [0] * len(segments)  # 모든 세그먼트를 같은 화자로 처리

    except Exception as e:
        log_message(f"❌ 화자 분석 오류: {e}")
        return [0] * len(segments)  # 에러 시 모든 세그먼트를 같은 화자로 처리


def analyze_speakers_pyannote(audio_path):
    """
    pyannote.audio를 사용한 고급 화자 분석
    
    Args:
        audio_path: 오디오 파일 경로
        
    Returns:
        speaker_timeline: [{'start': 0, 'end': 3200, 'speaker': 'SPEAKER_00'}, ...]
    """
    try:
        from pyannote.audio import Pipeline
        import torch

        log_message("🎤 pyannote.audio로 화자 분석 시작...")

        # 화자 분리 파이프라인 로드
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR_HF_TOKEN"  # Hugging Face 토큰 필요
        )

        # GPU 사용 가능시 GPU로
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))

        # 화자 분리 실행
        diarization = pipeline(audio_path)

        # 결과 정리
        speaker_timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_timeline.append({
                'start': turn.start * 1000,  # ms로 변환
                'end': turn.end * 1000,
                'speaker': speaker
            })

        num_speakers = len(set([s['speaker'] for s in speaker_timeline]))
        log_message(f"✅ pyannote 화자 분석 완료: {num_speakers}명의 화자 감지")

        return speaker_timeline

    except ImportError:
        log_message("❌ pyannote.audio가 설치되지 않았습니다.")
        log_message("pip install pyannote.audio 으로 설치해주세요.")
        return None

    except Exception as e:
        log_message(f"❌ pyannote 화자 분석 오류: {e}")
        return None


def smart_merge_by_speaker(segments, speaker_labels, min_duration_ms=3000):
    """
    화자별로 스마트하게 세그먼트 병합
    
    Args:
        segments: [(start_ms, end_ms), ...] 세그먼트 리스트
        speaker_labels: [0, 0, 1, 1, 0, ...] 화자 레이블
        min_duration_ms: 최소 길이 (기본 3초)
        
    Returns:
        merged_segments: 병합된 세그먼트 리스트
        merge_map: 원본 인덱스 → 병합된 인덱스 매핑
        merged_speaker_labels: 병합된 세그먼트의 화자 레이블
    """
    if len(segments) != len(speaker_labels):
        log_message("❌ 세그먼트와 화자 레이블 수가 맞지 않습니다.")
        return segments, {i: i for i in range(len(segments))}, speaker_labels

    merged_segments = []
    merged_speaker_labels = []
    merge_map = {}

    i = 0
    while i < len(segments):
        current_speaker = speaker_labels[i]
        group_segments = [segments[i]]
        group_indices = [i]

        # 같은 화자의 연속된 세그먼트들을 그룹화
        j = i + 1
        while j < len(segments) and speaker_labels[j] == current_speaker:
            group_segments.append(segments[j])
            group_indices.append(j)
            j += 1

        # 그룹 내에서 3초 이상 만들기
        merged_group, group_merge_map = merge_group_segments(
            group_segments, group_indices, min_duration_ms
        )

        # 결과에 추가
        start_merged_idx = len(merged_segments)
        merged_segments.extend(merged_group)
        merged_speaker_labels.extend([current_speaker] * len(merged_group))

        # 매핑 정보 업데이트
        for orig_idx, local_merged_idx in group_merge_map.items():
            merge_map[orig_idx] = start_merged_idx + local_merged_idx

        log_message(f"화자 {current_speaker}: {len(group_segments)}개 → {len(merged_group)}개 세그먼트")

        i = j

    log_message(f"✅ 화자별 스마트 병합 완료: {len(segments)}개 → {len(merged_segments)}개")
    return merged_segments, merge_map, merged_speaker_labels


def merge_group_segments(group_segments, group_indices, min_duration_ms):
    """같은 화자 그룹 내에서 세그먼트 병합"""
    if not group_segments:
        return [], {}

    merged = []
    merge_map = {}
    current_group = []
    current_indices = []

    for i, (start, end) in enumerate(group_segments):
        current_group.append((start, end))
        current_indices.append(group_indices[i])

        # 현재 그룹의 총 길이
        group_start = current_group[0][0]
        group_end = current_group[-1][1]
        total_duration = group_end - group_start

        # 3초 이상이 되었거나 마지막인 경우
        if total_duration >= min_duration_ms or i == len(group_segments) - 1:
            merged.append((group_start, group_end))
            merged_idx = len(merged) - 1

            # 매핑 정보
            for orig_idx in current_indices:
                merge_map[orig_idx] = merged_idx

            # 리셋
            current_group = []
            current_indices = []

    return merged, merge_map
