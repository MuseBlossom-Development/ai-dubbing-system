import os
import re
import unicodedata
from pydub import AudioSegment
from utils import log_message, audio_log_message

# SRT 파싱용 정규식
_time_re = re.compile(r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})')


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    파일명을 안전한 ASCII 문자로 변환하여 인코딩 문제 방지
    
    Args:
        filename: 원본 파일명
        max_length: 최대 길이 제한
    
    Returns:
        안전하게 변환된 파일명
    """
    if not filename:
        return "unnamed"

    # 파일 확장자 분리
    name, ext = os.path.splitext(filename)

    # 1단계: 유니코드 정규화 (한국어 → 라틴문자 변환 시도)
    try:
        # NFD로 정규화하여 분해된 문자들을 처리
        normalized = unicodedata.normalize('NFD', name)
        # ASCII가 아닌 문자들을 제거하고 ASCII만 남김
        ascii_name = ''.join(c for c in normalized if ord(c) < 128)
    except Exception:
        ascii_name = name

    # 2단계: 한국어 특수 처리 (일반적인 한국어 단어들을 영어로 변환)
    korean_to_english = {
        '음성': 'voice',
        '오디오': 'audio',
        '비디오': 'video',
        '영상': 'video',
        '파일': 'file',
        '테스트': 'test',
        '샘플': 'sample',
        '녹음': 'record',
        '변환': 'convert',
        '번역': 'translate',
        '합성': 'synthesis',
        '분할': 'split',
        '세그먼트': 'segment'
    }

    # 한국어가 포함된 경우 영어로 치환 시도
    result_name = ascii_name
    if not ascii_name or len(ascii_name.strip()) < 2:
        # ASCII 변환이 실패한 경우 한국어→영어 매핑 사용
        for korean, english in korean_to_english.items():
            if korean in name:
                result_name = name.replace(korean, english)
                break

        # 여전히 한국어가 남아있으면 타임스탬프 기반 이름 생성
        if not result_name or any(ord(c) > 127 for c in result_name):
            import time
            timestamp = int(time.time())
            result_name = f"audio_{timestamp}"

    # 3단계: 파일명 안전화
    # 특수문자 제거 및 공백을 언더스코어로 변환
    safe_name = re.sub(r'[^\w\-_.]', '_', result_name)
    safe_name = re.sub(r'_+', '_', safe_name)  # 연속 언더스코어 정리
    safe_name = safe_name.strip('_')  # 앞뒤 언더스코어 제거

    # 4단계: 길이 제한
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length].rstrip('_')

    # 5단계: 빈 이름 처리
    if not safe_name:
        safe_name = "unnamed"

    # 최종 파일명 조합
    final_name = safe_name + ext.lower()

    # 로그 출력 (변환이 발생한 경우만)
    if final_name != filename:
        log_message(f"🔤 파일명 안전화: '{filename}' → '{final_name}'")

    return final_name


def safe_file_operations(file_path: str, operation: str = "read") -> str:
    """
    파일 경로의 인코딩 문제를 방지하는 안전한 파일 작업
    
    Args:
        file_path: 파일 경로
        operation: 작업 유형 ("read", "write", "check")
    
    Returns:
        안전한 파일 경로 또는 오류 메시지
    """
    try:
        # 경로 정규화
        normalized_path = os.path.normpath(file_path)

        # 절대 경로로 변환
        abs_path = os.path.abspath(normalized_path)

        # 경로가 실제로 존재하는지 확인 (operation이 read나 check인 경우)
        if operation in ["read", "check"]:
            if not os.path.exists(abs_path):
                return f"❌ 파일이 존재하지 않음: {file_path}"

        # 디렉토리 생성 (operation이 write인 경우)
        if operation == "write":
            dir_path = os.path.dirname(abs_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                log_message(f"📁 디렉토리 생성: {dir_path}")

        return abs_path

    except Exception as e:
        error_msg = f"❌ 파일 경로 처리 오류 ({file_path}): {e}"
        log_message(error_msg)
        return error_msg


def srt_time_to_milliseconds(t: str) -> int:
    """SRT 시간 형식을 밀리초로 변환"""
    t = t.replace(',', '.')
    h, m, rest = t.split(':')
    s, ms = rest.split('.')
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def parse_srt_segments(srt_path: str):
    """SRT 파일에서 타임스탬프 세그먼트들을 파싱"""
    segments = []
    log_message(f"🔍 SRT 파싱 시작: {srt_path}")

    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        log_message(f"📄 SRT 파일 내용 (처음 500자): {content[:500]}...")

        # 파일을 다시 처음부터 읽기
        f.seek(0)
        line_num = 0
        for line in f:
            line_num += 1
            m = _time_re.search(line)
            if m:
                start, end = m.group(1), m.group(2)
                start_ms = srt_time_to_milliseconds(start)
                end_ms = srt_time_to_milliseconds(end)
                segments.append((start_ms, end_ms))
                log_message(f"🎯 세그먼트 {len(segments)}: {start} → {end} ({start_ms}~{end_ms}ms)")

    log_message(f"✅ SRT 파싱 완료: {len(segments)}개 세그먼트")
    return segments


def split_segments_by_speaker_changes(srt_segments, diarization_timeline):
    """
    화자 변경 지점에서 SRT 세그먼트를 분할
    
    Args:
        srt_segments: [(start_ms, end_ms), ...] 형태의 SRT 세그먼트
        diarization_timeline: [{'start': ms, 'end': ms, 'speaker': id}, ...] 형태의 화자 분리 결과
    
    Returns:
        split_segments: 화자 변경 지점에서 분할된 세그먼트 리스트
    """
    if not diarization_timeline:
        log_message("화자 분리 데이터가 없어 원본 세그먼트 유지")
        return srt_segments

    split_segments = []
    log_message(f"🎭 화자 변경 지점 분할 시작: {len(srt_segments)}개 세그먼트")

    for srt_idx, (srt_start, srt_end) in enumerate(srt_segments, 1):
        log_message(f"📝 SRT 세그먼트 {srt_idx}: {srt_start}~{srt_end}ms")

        # 이 SRT 구간과 겹치는 화자 구간들 찾기
        overlapping_speakers = []
        for dia_seg in diarization_timeline:
            dia_start = int(dia_seg['start'] * 1000)  # 초 → 밀리초
            dia_end = int(dia_seg['end'] * 1000)

            # 겹침 검사
            if dia_start < srt_end and dia_end > srt_start:
                overlap_start = max(srt_start, dia_start)
                overlap_end = min(srt_end, dia_end)

                overlapping_speakers.append({
                    'speaker': dia_seg['speaker'],
                    'start': overlap_start,
                    'end': overlap_end,
                    'duration': overlap_end - overlap_start
                })

        if not overlapping_speakers:
            # 화자 정보가 없으면 원본 유지
            split_segments.append((srt_start, srt_end))
            log_message(f"  → 화자 정보 없음, 원본 유지")
            continue

        # 시간순 정렬
        overlapping_speakers.sort(key=lambda x: x['start'])

        # 연속된 같은 화자 구간 병합
        merged_speakers = []
        for speaker_seg in overlapping_speakers:
            if merged_speakers and merged_speakers[-1]['speaker'] == speaker_seg['speaker']:
                # 같은 화자면 구간 확장
                merged_speakers[-1]['end'] = speaker_seg['end']
                merged_speakers[-1]['duration'] = merged_speakers[-1]['end'] - merged_speakers[-1]['start']
            else:
                # 다른 화자면 새 구간 추가
                merged_speakers.append(speaker_seg)

        # 화자별로 세그먼트 분할
        for i, speaker_seg in enumerate(merged_speakers):
            # 최소 길이 체크 (300ms 미만은 제외)
            if speaker_seg['duration'] < 300:
                log_message(f"  → 화자 {speaker_seg['speaker']}: {speaker_seg['duration']}ms (너무 짧아서 제외)")
                continue

            split_segments.append((speaker_seg['start'], speaker_seg['end']))
            log_message(
                f"  → 화자 {speaker_seg['speaker']}: {speaker_seg['start']}~{speaker_seg['end']}ms ({speaker_seg['duration']}ms)")

    log_message(f"✨ 화자 분할 완료: {len(srt_segments)}개 → {len(split_segments)}개 세그먼트")
    return split_segments


def apply_speaker_based_splitting(audio_path, srt_path, output_dir, enable_speaker_splitting=False):
    """
    화자 변경 지점 기반 오디오 분할
    
    Args:
        audio_path: 원본 오디오 파일 경로
        srt_path: SRT 파일 경로  
        output_dir: 출력 디렉토리
        enable_speaker_splitting: 화자 기반 분할 활성화 여부
    
    Returns:
        segments, total_audio_length: 분할된 세그먼트와 총 길이
    """
    # 기본 SRT 파싱
    original_segments = parse_srt_segments(srt_path)

    if not enable_speaker_splitting:
        log_message("화자 기반 분할 비활성화 - 기본 SRT 분할 사용")
        return split_audio_by_srt(audio_path, srt_path, output_dir)

    # 화자 분리 결과 파일 찾기
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    diarization_dir = os.path.join(os.getcwd(), 'speaker_analysis', base_name)
    diarization_report = os.path.join(diarization_dir, 'diarization_report.json')

    if not os.path.exists(diarization_report):
        log_message(f"화자 분리 결과 없음: {diarization_report}")
        log_message("기본 SRT 분할 사용 (먼저 '화자 분리 실행' 버튼을 눌러주세요)")
        return split_audio_by_srt(audio_path, srt_path, output_dir)

    try:
        # 화자 분리 결과 로드
        import json
        with open(diarization_report, 'r', encoding='utf-8') as f:
            diarization_data = json.load(f)

        diarization_timeline = diarization_data.get('timeline', [])
        log_message(f"📊 화자 분리 데이터 로드: {len(diarization_timeline)}개 구간")

        # 화자 변경 지점에서 세그먼트 분할
        split_segments = split_segments_by_speaker_changes(original_segments, diarization_timeline)

        # 분할된 세그먼트로 오디오 분할
        audio = AudioSegment.from_file(audio_path)
        total_audio_length = len(audio)

        wav_folder = os.path.join(output_dir, 'wav')
        base = os.path.splitext(os.path.basename(audio_path))[0]
        os.makedirs(wav_folder, exist_ok=True)

        log_message(f'🎭 화자 기반 오디오 분할 시작: {len(split_segments)}개 세그먼트')

        for idx, (start_ms, end_ms) in enumerate(split_segments, 1):
            duration = end_ms - start_ms

            # 세그먼트가 오디오 길이를 초과하는지 체크
            if start_ms >= total_audio_length:
                continue
            if end_ms > total_audio_length:
                end_ms = total_audio_length

            chunk = audio[start_ms:end_ms]
            out_path = os.path.join(wav_folder, f"{base}_{idx:03d}.wav")
            chunk.export(out_path, format="wav")

            actual_duration = len(chunk)
            log_message(f"✂️ 세그먼트 {idx}: {start_ms}~{end_ms}ms (목표:{duration}ms, 실제:{actual_duration}ms)")
            audio_log_message(f"세그먼트 {idx}: {start_ms}~{end_ms}")

        return split_segments, total_audio_length

    except Exception as e:
        log_message(f"화자 기반 분할 오류: {e}")
        log_message("기본 SRT 분할로 대체")
        return split_audio_by_srt(audio_path, srt_path, output_dir)


def process_individual_segments_for_synthesis(segments, min_duration_ms=500):
    """
    개별 세그먼트 처리 (3초 제약 없이 모든 세그먼트를 독립적으로 처리)
    
    Args:
        segments: [(start_ms, end_ms), ...] 형태의 세그먼트 리스트
        min_duration_ms: 처리할 최소 길이 (너무 짧은 세그먼트 필터링용)
    
    Returns:
        filtered_segments: 처리할 세그먼트 리스트
        segment_map: 원본 인덱스 → 필터링된 인덱스 매핑
    """
    if not segments:
        return [], {}

    filtered_segments = []
    segment_map = {}  # 원본 idx -> 필터링된 idx

    for i, (start_ms, end_ms) in enumerate(segments, start=1):
        duration = end_ms - start_ms

        # 최소 길이 체크 (너무 짧은 세그먼트는 제외)
        if duration >= min_duration_ms:
            filtered_segments.append((start_ms, end_ms))
            segment_map[i] = len(filtered_segments) - 1
            log_message(f"세그먼트 {i}: {duration}ms (처리 대상)")
        else:
            log_message(f"세그먼트 {i}: {duration}ms (너무 짧아서 제외)")

    log_message(f"✅ 개별 세그먼트 처리: {len(segments)}개 → {len(filtered_segments)}개")
    log_message("📍 모든 세그먼트가 개별적으로 정확한 타임라인에 배치됩니다")

    return filtered_segments, segment_map


def split_audio_by_srt(audio_path: str, srt_path: str, output_dir: str):
    """SRT 파일에 따라 오디오를 세그먼트로 분할"""
    log_message(f'🎵 오디오 분할 시작')
    log_message(f'   오디오: {audio_path}')
    log_message(f'   SRT: {srt_path}')
    log_message(f'   출력: {output_dir}')
    log_message(f'   SRT 존재: {os.path.exists(srt_path)}')

    segments = parse_srt_segments(srt_path)
    log_message(f'📊 파싱된 세그먼트 수: {len(segments)}')

    audio = AudioSegment.from_file(audio_path)
    total_audio_length = len(audio)
    log_message(f'🎼 원본 오디오 길이: {total_audio_length}ms ({total_audio_length / 1000:.1f}초)')

    wav_folder = os.path.join(output_dir, 'wav')
    base = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(wav_folder, exist_ok=True)

    log_message(f'📁 WAV 출력 폴더: {wav_folder}')
    log_message(f'📝 베이스 이름: {base}')

    for idx, (start_ms, end_ms) in enumerate(segments, 1):
        duration = end_ms - start_ms

        # 세그먼트가 오디오 길이를 초과하는지 체크
        if start_ms >= total_audio_length:
            log_message(f"⚠️ 세그먼트 {idx}: 시작시간({start_ms}ms)이 오디오 길이({total_audio_length}ms)를 초과 - 건너뛰기")
            continue

        if end_ms > total_audio_length:
            log_message(f"⚠️ 세그먼트 {idx}: 종료시간({end_ms}ms)이 오디오 길이를 초과 - {total_audio_length}ms로 조정")
            end_ms = total_audio_length

        chunk = audio[start_ms:end_ms]
        out_path = os.path.join(wav_folder, f"{base}_{idx:03d}.wav")
        chunk.export(out_path, format="wav")

        actual_duration = len(chunk)
        log_message(f"✂️ 세그먼트 {idx}: {start_ms}~{end_ms}ms (목표:{duration}ms, 실제:{actual_duration}ms)")
        audio_log_message(f"세그먼트 {idx}: {start_ms}~{end_ms}")

    return segments, total_audio_length


def extend_short_segments_for_zeroshot(segments_dir, min_duration_ms=3000):
    """
    3초 미만 세그먼트를 복사 붙여넣기로 3초 이상으로 확장
    제로샷 음성 합성을 위한 전처리
    
    Args:
        segments_dir: wav 세그먼트들이 있는 디렉토리
        min_duration_ms: 최소 길이 (기본 3초 = 3000ms)
    
    Returns:
        extended_segments_dir: 확장된 세그먼트들이 저장된 디렉토리
    """
    wav_dir = os.path.join(segments_dir, 'wav')
    if not os.path.exists(wav_dir):
        log_message(f"❌ WAV 디렉토리 없음: {wav_dir}")
        return None

    # 확장된 세그먼트 저장 디렉토리 생성
    extended_dir = os.path.join(segments_dir, 'wav_extended_3sec')
    os.makedirs(extended_dir, exist_ok=True)

    log_message(f"🔄 3초 미만 세그먼트 확장 시작 (최소 {min_duration_ms}ms)")
    log_message(f"📁 원본: {wav_dir}")
    log_message(f"📁 확장본: {extended_dir}")

    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    wav_files.sort()  # 파일명 순서대로 정렬

    extended_count = 0
    copied_count = 0

    for wav_file in wav_files:
        input_path = os.path.join(wav_dir, wav_file)
        output_path = os.path.join(extended_dir, wav_file)

        try:
            # 원본 세그먼트 로드
            audio = AudioSegment.from_file(input_path)
            original_duration = len(audio)

            if original_duration >= min_duration_ms:
                # 이미 3초 이상이면 그대로 복사
                audio.export(output_path, format="wav")
                copied_count += 1
                log_message(f"✅ {wav_file}: {original_duration}ms (복사)")
            else:
                # 3초 미만이면 반복해서 확장
                extended_audio = extend_audio_by_repetition(audio, min_duration_ms)
                extended_audio.export(output_path, format="wav")
                extended_count += 1

                final_duration = len(extended_audio)
                repetitions = final_duration // original_duration
                log_message(f"🔄 {wav_file}: {original_duration}ms → {final_duration}ms ({repetitions}회 반복)")

        except Exception as e:
            log_message(f"❌ {wav_file} 처리 실패: {e}")
            continue

    log_message(f"✨ 세그먼트 확장 완료!")
    log_message(f"📊 확장된 파일: {extended_count}개")
    log_message(f"📊 복사된 파일: {copied_count}개")
    log_message(f"📊 총 파일: {extended_count + copied_count}개")
    log_message(f"💾 저장 위치: {extended_dir}")

    return extended_dir


def extend_audio_by_repetition(audio_segment, target_duration_ms):
    """
    오디오를 반복해서 목표 길이까지 확장
    자연스러운 연결을 위해 페이드 처리 적용
    
    Args:
        audio_segment: 원본 오디오 세그먼트
        target_duration_ms: 목표 길이 (밀리초)
    
    Returns:
        확장된 오디오 세그먼트
    """
    if len(audio_segment) >= target_duration_ms:
        return audio_segment

    original_duration = len(audio_segment)
    extended_audio = AudioSegment.empty()

    # 짧은 페이드 시간 계산 (원본의 5% 또는 최대 100ms)
    fade_duration = min(int(original_duration * 0.05), 100)

    current_duration = 0
    repetition_count = 0

    while current_duration < target_duration_ms:
        remaining_duration = target_duration_ms - current_duration

        if remaining_duration >= original_duration:
            # 전체 세그먼트 추가
            if repetition_count == 0:
                # 첫 번째 반복은 그대로
                segment_to_add = audio_segment
            else:
                # 이후 반복은 페이드인으로 자연스럽게 연결
                segment_to_add = audio_segment.fade_in(fade_duration)

            extended_audio += segment_to_add
            current_duration += len(segment_to_add)
            repetition_count += 1

        else:
            # 부분 세그먼트 추가 (마지막 조각)
            partial_segment = audio_segment[:remaining_duration]
            if repetition_count > 0:
                partial_segment = partial_segment.fade_in(fade_duration)
            partial_segment = partial_segment.fade_out(fade_duration)

            extended_audio += partial_segment
            current_duration += len(partial_segment)
            break

    return extended_audio


def create_extended_segments_mapping(original_segments_dir, extended_segments_dir):
    """
    원본과 확장된 세그먼트 간의 매핑 정보 생성
    
    Args:
        original_segments_dir: 원본 세그먼트 디렉토리
        extended_segments_dir: 확장된 세그먼트 디렉토리
    
    Returns:
        매핑 정보가 담긴 딕셔너리
    """
    import json

    original_wav_dir = os.path.join(original_segments_dir, 'wav')
    mapping_info = {
        'original_dir': original_wav_dir,
        'extended_dir': extended_segments_dir,
        'segments_info': [],
        'created_at': str(os.path.getctime(extended_segments_dir))
    }

    if not os.path.exists(original_wav_dir) or not os.path.exists(extended_segments_dir):
        return mapping_info

    original_files = [f for f in os.listdir(original_wav_dir) if f.endswith('.wav')]
    extended_files = [f for f in os.listdir(extended_segments_dir) if f.endswith('.wav')]

    for wav_file in sorted(original_files):
        if wav_file in extended_files:
            original_path = os.path.join(original_wav_dir, wav_file)
            extended_path = os.path.join(extended_segments_dir, wav_file)

            try:
                original_audio = AudioSegment.from_file(original_path)
                extended_audio = AudioSegment.from_file(extended_path)

                segment_info = {
                    'filename': wav_file,
                    'original_duration_ms': len(original_audio),
                    'extended_duration_ms': len(extended_audio),
                    'was_extended': len(extended_audio) > len(original_audio),
                    'repetition_ratio': len(extended_audio) / len(original_audio) if len(original_audio) > 0 else 1
                }

                mapping_info['segments_info'].append(segment_info)

            except Exception as e:
                log_message(f"매핑 정보 생성 실패 ({wav_file}): {e}")

    # 매핑 정보를 JSON 파일로 저장
    mapping_file = os.path.join(extended_segments_dir, 'extension_mapping.json')
    try:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_info, f, indent=2, ensure_ascii=False)
        log_message(f"📄 매핑 정보 저장: {mapping_file}")
    except Exception as e:
        log_message(f"매핑 정보 저장 실패: {e}")

    return mapping_info


def adjust_audio_speed(audio_segment, speed_factor):
    """오디오 속도를 조절하면서 피치 보존"""
    try:
        # 속도 조절 (프레임 레이트 변경)
        adjusted = audio_segment._spawn(
            audio_segment.raw_data,
            overrides={"frame_rate": int(audio_segment.frame_rate * speed_factor)}
        ).set_frame_rate(audio_segment.frame_rate)
        return adjusted
    except Exception as e:
        log_message(f"속도 조절 오류: {e}")
        return audio_segment


def calculate_segment_priority(text_content, duration_ms):
    """세그먼트의 중요도 점수 계산"""
    if not text_content:
        return 0.1  # 텍스트가 없으면 낮은 우선순위

    # 기본 점수
    base_score = 0.5

    # 길이 기반 보정 (너무 짧거나 긴 것은 우선순위 낮춤)
    if duration_ms < 500:  # 0.5초 미만
        base_score *= 0.7
    elif duration_ms > 10000:  # 10초 초과
        base_score *= 0.8

    # 문장 부호 기반 중요도 (완전한 문장은 높은 우선순위)
    if any(punct in text_content for punct in ['.', '!', '?', '다', '요', '니다']):
        base_score *= 1.2

    # 짧은 감탄사나 응답은 우선순위 낮춤
    if len(text_content) <= 3 and text_content in ['네', '예', '아', '오', '응', '음']:
        base_score *= 0.6

    return min(1.0, max(0.1, base_score))


def trim_leading_silence(audio_segment, silence_thresh_offset=-40):
    """
    오디오 앞부분의 무음 구간을 제거
    
    Args:
        audio_segment: 처리할 오디오 세그먼트
        silence_thresh_offset: 무음 임계값 (dBFS 기준, 기본 -40dB)
    
    Returns:
        앞부분 무음이 제거된 오디오 세그먼트
    """
    try:
        if len(audio_segment) == 0:
            return audio_segment

        # 무음 임계값 설정 (평균 볼륨에서 offset만큼 낮은 값)
        silence_thresh = audio_segment.dBFS + silence_thresh_offset

        # 50ms 단위로 검사하여 무음 구간 찾기
        chunk_size = 50
        trim_start = 0

        for i in range(0, len(audio_segment), chunk_size):
            chunk = audio_segment[i:i + chunk_size]

            # 청크가 너무 짧으면 건너뛰기
            if len(chunk) < chunk_size // 2:
                break

            # 무음이 아니면 중단
            if chunk.dBFS > silence_thresh:
                break

            trim_start = i + chunk_size

        # 최대 2초까지만 제거 (너무 많이 제거되는 것 방지)
        max_trim = min(trim_start, 2000)

        if max_trim > 0:
            trimmed = audio_segment[max_trim:]
            log_message(f"🔇 앞부분 무음 제거: {max_trim}ms")
            return trimmed
        else:
            return audio_segment

    except Exception as e:
        log_message(f"앞부분 무음 제거 오류: {e}")
        return audio_segment


def simple_speed_adjustment(audio_segment, target_duration_ms):
    """
    CosyVoice speed 파라미터 사용 시 후처리 배속 조절 최소화
    주로 무음 제거와 미세 조정에만 사용
    
    Args:
        audio_segment: 합성된 오디오 세그먼트
        target_duration_ms: 목표 길이 (원본 길이)
    
    Returns:
        조정된 오디오 세그먼트
    """
    # 1단계: 앞부분 무음 제거
    trimmed_audio = trim_leading_silence(audio_segment)
    current_duration = len(trimmed_audio)

    # 2단계: 길이 체크 (더 관대한 허용 범위)
    tolerance = max(200, target_duration_ms * 0.1)  # 200ms 또는 10% 중 큰 값
    if abs(current_duration - target_duration_ms) <= tolerance:
        log_message(f"길이 차이 허용 범위 내: {abs(current_duration - target_duration_ms)}ms")
        return trimmed_audio

    # 3단계: 미세 조정만 수행 (CosyVoice에서 이미 속도 조절했으므로)
    duration_diff = current_duration - target_duration_ms

    if duration_diff > tolerance:
        # 길이가 너무 길면 자연스럽게 트림
        fade_duration = min(200, target_duration_ms // 20)
        adjusted = trimmed_audio[:target_duration_ms].fade_out(fade_duration)
        log_message(f"후처리 트림: {current_duration}ms → {len(adjusted)}ms")
        return adjusted
    elif duration_diff < -tolerance:
        # 길이가 부족하면 자연스러운 패딩
        padding_duration = -duration_diff
        padding = AudioSegment.silent(duration=int(padding_duration))
        adjusted = trimmed_audio + padding
        log_message(f"후처리 패딩: {current_duration}ms → {len(adjusted)}ms")
        return adjusted

    return trimmed_audio


def remove_excessive_silence(audio_segment, max_silence_ms=500):
    """과도한 무음 구간 제거"""
    try:
        # 무음 임계값 설정 (-40dB)
        silence_thresh = audio_segment.dBFS - 40

        # 무음 구간 탐지
        chunks = []
        current_pos = 0
        chunk_size = 100  # 100ms 단위로 처리

        while current_pos < len(audio_segment):
            chunk = audio_segment[current_pos:current_pos + chunk_size]

            # 무음 여부 확인
            if chunk.dBFS < silence_thresh:
                # 무음 구간 - 길이 제한
                silence_length = min(chunk_size, max_silence_ms)
                chunks.append(AudioSegment.silent(duration=silence_length))

                # 연속 무음 건너뛰기
                next_pos = current_pos + chunk_size
                while next_pos < len(audio_segment):
                    next_chunk = audio_segment[next_pos:next_pos + chunk_size]
                    if next_chunk.dBFS >= silence_thresh:
                        break
                    next_pos += chunk_size
                current_pos = next_pos
            else:
                # 음성 구간 - 그대로 유지
                chunks.append(chunk)
                current_pos += chunk_size

        return sum(chunks) if chunks else audio_segment

    except Exception as e:
        log_message(f"무음 제거 오류: {e}")
        return audio_segment


def smart_audio_compression(audio_segment, target_duration_ms, text_content=""):
    """CosyVoice speed 사용 시 간단한 후처리만 수행"""
    log_message("CosyVoice speed 파라미터로 이미 길이 조절됨 - 최소 후처리만 적용")
    return simple_speed_adjustment(audio_segment, target_duration_ms)


def merge_segments_preserve_timing(segments, original_duration_ms, segments_dir, output_path,
                                   length_handling="preserve", overlap_handling="fade", max_extension=50,
                                   enable_smart_compression=True, correct_cosyvoice_padding=True):
    """세그먼트들을 원본 타임라인에 맞춰 정확히 병합 (절대 위치 기반)
    
    Args:
        correct_cosyvoice_padding: CosyVoice 패딩(0.2초) 보정 여부
    """
    # 안전장치: 입력값 검증
    if not segments:
        log_message("❌ 세그먼트 데이터가 없습니다")
        return original_duration_ms

    if original_duration_ms <= 0:
        log_message("❌ 원본 길이가 유효하지 않습니다")
        return 0

    # 안전장치: 세그먼트 타임스탬프 검증
    max_reasonable_duration = 3600000  # 1시간 = 3,600,000ms
    if original_duration_ms > max_reasonable_duration:
        log_message(f"⚠️ 원본 길이가 비정상적으로 큼: {original_duration_ms}ms ({original_duration_ms / 60000:.1f}분)")
        original_duration_ms = min(original_duration_ms, max_reasonable_duration)
        log_message(f"🔧 안전한 길이로 제한: {original_duration_ms}ms")

    # 세그먼트 유효성 검사
    valid_segments = []
    for i, (start_ms, end_ms) in enumerate(segments, 1):
        if start_ms < 0 or end_ms < 0:
            log_message(f"⚠️ 세그먼트 {i}: 음수 타임스탬프 무시 ({start_ms}, {end_ms})")
            continue

        if end_ms <= start_ms:
            log_message(f"⚠️ 세그먼트 {i}: 유효하지 않은 구간 무시 ({start_ms}, {end_ms})")
            continue

        if start_ms > max_reasonable_duration or end_ms > max_reasonable_duration:
            log_message(f"⚠️ 세그먼트 {i}: 비정상적으로 큰 타임스탬프 무시 ({start_ms}, {end_ms})")
            continue

        duration = end_ms - start_ms
        if duration > 600000:  # 10분 초과
            log_message(f"⚠️ 세그먼트 {i}: 과도하게 긴 구간 무시 ({duration}ms)")
            continue

        valid_segments.append((start_ms, end_ms))

    if not valid_segments:
        log_message("❌ 유효한 세그먼트가 없습니다")
        return original_duration_ms

    log_message(f"✅ 유효한 세그먼트: {len(valid_segments)}/{len(segments)}개")
    segments = valid_segments

    # 베이스 이름 추출
    if 'cosy_output' in segments_dir:
        path_parts = segments_dir.split(os.sep)
        cosy_index = None
        for i, part in enumerate(path_parts):
            if part == 'cosy_output':
                cosy_index = i
                break

        if cosy_index and cosy_index >= 2:
            split_audio_index = None
            for i, part in enumerate(path_parts):
                if part == 'split_audio':
                    split_audio_index = i
                    break

            if split_audio_index is not None and split_audio_index + 1 < len(path_parts):
                base = path_parts[split_audio_index + 1]
            else:
                base = os.path.basename(os.path.dirname(os.path.dirname(segments_dir)))
        else:
            base = os.path.basename(segments_dir)
    else:
        base = os.path.basename(segments_dir)

    # CosyVoice 패딩 보정값 설정 (0.2초 = 200ms)
    padding_correction_ms = 200 if correct_cosyvoice_padding else 0

    log_message(f"🎯 절대 위치 기반 타임라인 병합 시작")
    log_message(f"베이스: {base}, 원본 길이: {original_duration_ms}ms")
    log_message(f"설정: 길이={length_handling}, 겹침={overlap_handling}, 최대확장={max_extension}%")
    if correct_cosyvoice_padding:
        log_message(f"🔧 CosyVoice 패딩 보정 활성화: 각 세그먼트가 {padding_correction_ms}ms 앞당겨짐")

    # 실제 존재하는 파일들 확인
    if not os.path.exists(segments_dir):
        log_message(f"❌ 세그먼트 디렉토리 없음: {segments_dir}")
        return original_duration_ms

    available_files = [f for f in os.listdir(segments_dir) if f.endswith('.wav')]
    log_message(f"사용 가능한 파일: {len(available_files)}개")

    # 1단계: 모든 세그먼트 로드 및 전처리
    processed_segments = []

    for idx, (start_ms, end_ms) in enumerate(segments, start=1):
        original_duration = end_ms - start_ms
        seg_path = os.path.join(segments_dir, f"{base}_{idx:03d}.wav")

        # CosyVoice 패딩 보정을 위한 실제 배치 위치 계산
        corrected_start_ms = max(0, start_ms - padding_correction_ms)

        segment_data = {
            'idx': idx,
            'original_start': start_ms,
            'corrected_start': corrected_start_ms,  # 패딩 보정된 시작 위치
            'original_end': end_ms,
            'original_duration': original_duration,
            'audio': None,
            'final_duration': original_duration,
            'exists': False
        }

        if os.path.exists(seg_path):
            try:
                # 합성 음성 로드
                synth_audio = AudioSegment.from_file(seg_path)
                synth_duration = len(synth_audio)

                # 안전장치: 합성 파일 크기 검증
                if synth_duration > 600000:  # 10분 초과
                    log_message(f"⚠️ 세그먼트 {idx}: 합성 파일이 과도하게 큼 ({synth_duration}ms) - 10분으로 제한")
                    synth_audio = synth_audio[:600000].fade_out(1000)
                    synth_duration = len(synth_audio)

                # 텍스트 내용 로드 (스마트 압축용)
                text_content = ""
                if enable_smart_compression:
                    text_file = os.path.join(segments_dir, 'txt', 'ko', f"{base}_{idx:03d}.ko.txt")
                    if os.path.exists(text_file):
                        try:
                            with open(text_file, 'r', encoding='utf-8') as f:
                                text_content = f.read().strip()
                        except Exception:
                            pass

                # 길이 처리 로직
                if length_handling == "preserve":
                    # 보존 모드: 합성 결과를 그대로 사용하되 극단적인 경우만 제한
                    if synth_duration > original_duration * 2:  # 2배 초과시만 제한
                        log_message(f"⚠️ 세그먼트 {idx}: 과도한 확장 제한 ({synth_duration}ms → {original_duration * 2}ms)")
                        synth_audio = synth_audio[:int(original_duration * 2)].fade_out(200)
                        synth_duration = len(synth_audio)

                    final_audio = synth_audio
                    final_duration = synth_duration

                elif length_handling == "fit":
                    # 맞춤 모드: 원본 길이에 최대한 맞춤
                    target_duration = original_duration

                    if enable_smart_compression and synth_duration > target_duration:
                        # 스마트 압축 적용
                        synth_audio = remove_excessive_silence(synth_audio, max_silence_ms=300)
                        if len(synth_audio) > target_duration:
                            synth_audio = smart_audio_compression(synth_audio, target_duration, text_content)

                    # 최종 길이 조정
                    current_duration = len(synth_audio)
                    if current_duration > target_duration:
                        # 자연스러운 컷
                        synth_audio = synth_audio[:target_duration].fade_out(min(200, target_duration // 10))
                    elif current_duration < target_duration:
                        # 무음 패딩
                        padding = AudioSegment.silent(duration=target_duration - current_duration)
                        synth_audio = synth_audio + padding

                    final_audio = synth_audio
                    final_duration = len(final_audio)

                segment_data.update({
                    'audio': final_audio,
                    'final_duration': final_duration,
                    'exists': True
                })

                if correct_cosyvoice_padding:
                    log_message(
                        f"✅ 세그먼트 {idx}: {original_duration}ms → {final_duration}ms (위치: {start_ms}ms → {corrected_start_ms}ms)")
                else:
                    log_message(f"✅ 세그먼트 {idx}: {original_duration}ms → {final_duration}ms")

            except Exception as e:
                log_message(f"❌ 세그먼트 {idx} 로드 실패: {e}")

        processed_segments.append(segment_data)

    # 2단계: 전체 타임라인 길이 계산 (안전장치 포함)
    if length_handling == "preserve":
        # 보존 모드: 가장 긴 세그먼트까지의 길이 (패딩 보정 고려)
        max_end_time = 0
        for seg in processed_segments:
            if seg['exists']:
                # 패딩 보정된 시작 위치 사용
                projected_end = seg['corrected_start'] + seg['final_duration']
                max_end_time = max(max_end_time, projected_end)

        final_timeline_length = max(original_duration_ms, max_end_time)

    else:  # fit 모드
        # 맞춤 모드: 원본 길이 기준 (최대 확장 제한)
        max_allowed_extension = original_duration_ms * max_extension / 100
        final_timeline_length = original_duration_ms + max_allowed_extension

    # 안전장치: 최종 타임라인 길이 제한
    if final_timeline_length > max_reasonable_duration:
        log_message(f"⚠️ 최종 타임라인이 과도하게 큼: {final_timeline_length}ms - 제한 적용")
        final_timeline_length = min(final_timeline_length, max_reasonable_duration)
        log_message(f"🔧 안전한 길이로 제한: {final_timeline_length}ms")

    log_message(f"📏 최종 타임라인 길이: {final_timeline_length}ms ({final_timeline_length / 60000:.1f}분)")

    # 안전장치: 메모리 사용량 체크 (대략적 계산)
    estimated_memory_mb = (final_timeline_length * 44100 * 2 * 2) / (1024 * 1024)  # 44.1kHz, 16bit, stereo
    if estimated_memory_mb > 1000:  # 1GB 초과
        log_message(f"⚠️ 예상 메모리 사용량이 과도함: {estimated_memory_mb:.1f}MB")
        # 더 작은 길이로 제한
        safe_length = min(final_timeline_length, 600000)  # 10분으로 제한
        log_message(f"🔧 메모리 안전을 위해 길이 제한: {safe_length}ms")
        final_timeline_length = safe_length

    # 3단계: 빈 타임라인 생성 (안전한 방법)
    try:
        log_message(f"💾 {final_timeline_length}ms 빈 타임라인 생성 중...")
        final_timeline = AudioSegment.silent(duration=int(final_timeline_length))
        log_message(f"✅ 타임라인 생성 완료: {len(final_timeline)}ms")
    except MemoryError:
        log_message("❌ 메모리 부족으로 타임라인 생성 실패 - 더 작은 크기로 재시도")
        final_timeline_length = min(final_timeline_length, 300000)  # 5분으로 축소
        final_timeline = AudioSegment.silent(duration=int(final_timeline_length))
        log_message(f"🔧 축소된 타임라인 생성: {len(final_timeline)}ms")
    except Exception as e:
        log_message(f"❌ 타임라인 생성 실패: {e}")
        return original_duration_ms

    # 4단계: 겹침 감지 및 해결 (보정된 위치 기준)
    overlap_pairs = []
    for i, seg1 in enumerate(processed_segments):
        if not seg1['exists']:
            continue

        seg1_start = seg1['corrected_start']  # 패딩 보정된 위치 사용
        seg1_end = seg1_start + seg1['final_duration']

        for j, seg2 in enumerate(processed_segments[i + 1:], i + 1):
            if not seg2['exists']:
                continue

            seg2_start = seg2['corrected_start']  # 패딩 보정된 위치 사용
            seg2_end = seg2_start + seg2['final_duration']

            # 겹침 검사
            if seg1_end > seg2_start and seg1_start < seg2_end:
                overlap_start = max(seg1_start, seg2_start)
                overlap_end = min(seg1_end, seg2_end)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration > 50:  # 50ms 이상 겹침만 처리
                    overlap_pairs.append({
                        'seg1_idx': i,
                        'seg2_idx': j,
                        'overlap_start': overlap_start,
                        'overlap_end': overlap_end,
                        'overlap_duration': overlap_duration
                    })

                    log_message(f"⚠️ 겹침 감지: 세그먼트 {seg1['idx']}-{seg2['idx']} ({overlap_duration}ms)")

    # 5단계: 겹침 해결 처리
    if overlap_pairs and overlap_handling == "fade":
        for overlap in overlap_pairs:
            seg1 = processed_segments[overlap['seg1_idx']]
            seg2 = processed_segments[overlap['seg2_idx']]

            # 크로스페이드 길이 계산 (겹침의 80% 또는 최대 500ms)
            fade_duration = min(int(overlap['overlap_duration'] * 0.8), 500)

            # 첫 번째 세그먼트 페이드아웃
            if seg1['audio'] and len(seg1['audio']) > fade_duration:
                seg1['audio'] = seg1['audio'].fade_out(fade_duration)
                log_message(f"🔧 세그먼트 {seg1['idx']}: {fade_duration}ms 페이드아웃")

            # 두 번째 세그먼트 페이드인
            if seg2['audio'] and len(seg2['audio']) > fade_duration:
                seg2['audio'] = seg2['audio'].fade_in(fade_duration)
                log_message(f"🔧 세그먼트 {seg2['idx']}: {fade_duration}ms 페이드인")

    elif overlap_pairs and overlap_handling == "cut":
        for overlap in overlap_pairs:
            seg1 = processed_segments[overlap['seg1_idx']]

            # 첫 번째 세그먼트를 겹치기 직전까지 자르기
            if seg1['audio']:
                cut_point = seg1['final_duration'] - overlap['overlap_duration']
                if cut_point > 0:
                    seg1['audio'] = seg1['audio'][:int(cut_point)].fade_out(100)
                    seg1['final_duration'] = len(seg1['audio'])
                    log_message(f"✂️ 세그먼트 {seg1['idx']}: {overlap['overlap_duration']}ms 컷")

    # 6단계: 절대 위치에 세그먼트 배치 (패딩 보정된 위치 사용)
    placement_successful = 0

    log_message(f"🎯 절대 위치 기반 세그먼트 배치 시작...")

    for seg in processed_segments:
        if seg['exists'] and seg['audio']:
            # 패딩 보정된 시작 위치 사용
            start_pos = seg['corrected_start']

            # 타임라인 범위 체크
            if start_pos < final_timeline_length:
                # 세그먼트가 타임라인을 벗어나지 않도록 조정
                available_length = final_timeline_length - start_pos
                if len(seg['audio']) > available_length:
                    seg['audio'] = seg['audio'][:int(available_length)].fade_out(100)

                try:
                    # 볼륨 정규화 - overlay 시 볼륨 손실 방지
                    normalized_segment = seg['audio']

                    # 볼륨이 너무 낮으면 증폭
                    if normalized_segment.dBFS < -30:
                        gain = -20 - normalized_segment.dBFS  # -20dBFS 목표
                        normalized_segment = normalized_segment + gain
                        log_message(f"  세그먼트 {seg['idx']} 볼륨 증폭: {gain:.1f}dB")

                    # overlay 시 gain_during_overlay 파라미터로 볼륨 손실 방지
                    final_timeline = final_timeline.overlay(
                        normalized_segment,
                        position=int(start_pos),
                        gain_during_overlay=0  # 볼륨 감소 없이 오버레이
                    )

                    placement_successful += 1
                    actual_end = start_pos + len(seg['audio'])

                    if correct_cosyvoice_padding:
                        correction_applied = seg['original_start'] - seg['corrected_start']
                        log_message(
                            f"🎯 세그먼트 {seg['idx']}: {start_pos}ms~{actual_end}ms 배치 완료 (보정: -{correction_applied}ms, dBFS: {normalized_segment.dBFS:.1f})")
                    else:
                        log_message(
                            f"🎯 세그먼트 {seg['idx']}: {start_pos}ms~{actual_end}ms 배치 완료 (dBFS: {normalized_segment.dBFS:.1f})")

                except Exception as e:
                    log_message(f"❌ 세그먼트 {seg['idx']} 배치 실패: {e}")

                    # 대체 방법: 수동으로 오디오 데이터 삽입
                    try:
                        # 원본 타임라인을 배열로 변환
                        timeline_samples = final_timeline.get_array_of_samples()
                        segment_samples = seg['audio'].get_array_of_samples()

                        # 시작 위치를 샘플 인덱스로 변환
                        sample_rate = final_timeline.frame_rate
                        start_sample = int(start_pos * sample_rate / 1000)

                        # 세그먼트 길이만큼 오버레이 (덧기)
                        for i, sample in enumerate(segment_samples):
                            if start_sample + i < len(timeline_samples):
                                # 기존 샘플과 새 샘플을 믹싱 (클리핑 방지)
                                mixed_sample = timeline_samples[start_sample + i] + sample
                                # 클리핑 방지 (16비트 범위)
                                timeline_samples[start_sample + i] = max(-32768, min(32767, mixed_sample))

                        # 배열을 다시 AudioSegment로 변환
                        final_timeline = final_timeline._spawn(timeline_samples)

                        placement_successful += 1
                        log_message(f"🔄 세그먼트 {seg['idx']}: 수동 믹싱 방식으로 배치 완료")

                    except Exception as e2:
                        log_message(f"❌ 세그먼트 {seg['idx']} 수동 믹싱도 실패: {e2}")

    log_message(f"📊 배치 성공: {placement_successful}/{len([s for s in processed_segments if s['exists']])} 세그먼트")

    # 최종 볼륨 검증
    if final_timeline.dBFS < -50:
        log_message(f"⚠️ 최종 결과 볼륨이 너무 낮음 ({final_timeline.dBFS:.1f}dBFS) - 증폭 적용")
        final_timeline = final_timeline + (max(-20, -10 - final_timeline.dBFS))  # -10dBFS 목표

    # 7단계: 최종 결과 저장
    actual_length = len(final_timeline)
    final_timeline.export(output_path, format="wav")

    log_message(f"🎵 절대 위치 기반 병합 완료!")
    log_message(f"📊 최종 길이: {actual_length}ms (원본: {original_duration_ms}ms)")
    log_message(f"📈 길이 변화: {actual_length - original_duration_ms:+d}ms")
    if correct_cosyvoice_padding:
        log_message(f"🔧 CosyVoice 패딩 보정 적용됨: 모든 세그먼트가 {padding_correction_ms}ms 앞당겨짐")
    log_message(f"💾 저장 완료: {output_path}")

    # 품질 검증
    if length_handling == "preserve":
        log_message("✅ 타임라인 보존 모드: 모든 합성 음성이 정확한 위치에 배치됨")
    else:
        extension_percent = ((actual_length - original_duration_ms) / original_duration_ms) * 100
        log_message(f"✅ 길이 맞춤 모드: {extension_percent:+.1f}% 변화")

    return actual_length
