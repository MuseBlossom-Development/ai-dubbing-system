import os
import sys
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading

# 환경 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 모듈화된 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CosyVoice'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CosyVoice', 'third_party'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CosyVoice', 'third_party', 'Matcha-TTS'))

from config import load_vad_config, save_vad_config, IS_WINDOWS
from utils import install_log_queue, audio_log_queue
from main_processor import start_processing_with_settings
from whisper_processor import generate_srt_only
from audio_processor import parse_srt_segments, merge_segments_preserve_timing
from speaker_diarization import test_speaker_diarization
from pydub import AudioSegment


def process_log_queue():
    """통합 로그 처리 함수"""
    while not install_log_queue.empty():
        line = install_log_queue.get()
        log_text.configure(state='normal')
        log_text.insert(tk.END, line + '\n')
        log_text.see(tk.END)
        log_text.configure(state='disabled')

    while not audio_log_queue.empty():
        line = audio_log_queue.get()
        log_text.configure(state='normal')
        log_text.insert(tk.END, f"[AUDIO] {line}" + '\n')
        log_text.see(tk.END)
        log_text.configure(state='disabled')

    root.after(200, process_log_queue)


def collect_gui_settings():
    """GUI에서 현재 설정들을 수집"""
    return {
        # Instruct2 설정
        'enable_instruct': enable_instruct_var.get(),
        'manual_command': manual_command_var.get() if command_mode_var.get() == 'manual' else None,

        # 타임라인 & 길이 처리
        'length_handling': length_handling_var.get(),
        'overlap_handling': overlap_handling_var.get(),
        'max_extension': int(max_extension_var.get()),

        # 번역 설정
        'selected_languages': [
            lang for lang, var in [
                ('english', enable_english_var),
                ('chinese', enable_chinese_var),
                ('japanese', enable_japanese_var)
            ] if var.get()
        ],
        'translation_length': float(translation_length_var.get()),
        'quality_mode': translation_quality_var.get(),

        # 스마트 압축
        'enable_smart_compression': enable_smart_compression_var.get(),

        # 화자 분리
        'enable_speaker_diarization': enable_speaker_diarization_var.get(),
        'num_speakers': int(num_speakers_var.get()) if speaker_mode_var.get() == 'fixed' else None,

        # 화자 기반 분할 설정
        'enable_speaker_splitting': enable_speaker_splitting_var.get(),

        # 3초 확장 설정
        'enable_3sec_extension': enable_3sec_extension_var.get(),

        # 음성 볼륨 (영상 처리용)
        'vocals_volume': 1.0,
        'background_volume': 0.8
    }


def start_processing():
    """메인 처리 시작 (영상/음성 자동 감지)"""
    # 플랫폼별 파일타입 형식 분기처리
    if IS_WINDOWS:
        filetypes = [
            ('All Media Files', '*.wav;*.mp3;*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv'),
            ('Audio Files', '*.wav;*.mp3'),
            ('Video Files', '*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv'),
            ('WAV files', '*.wav'),
            ('MP3 files', '*.mp3'),
            ('MP4 files', '*.mp4'),
            ('All files', '*.*')
        ]
    else:
        filetypes = [
            ('All Media Files', '*.wav *.mp3 *.mp4 *.avi *.mkv *.mov *.wmv *.flv'),
            ('Audio Files', '*.wav *.mp3'),
            ('Video Files', '*.mp4 *.avi *.mkv *.mov *.wmv *.flv'),
            ('WAV files', '*.wav'),
            ('MP3 files', '*.mp3'),
            ('MP4 files', '*.mp4'),
            ('All files', '*.*')
        ]

    input_file = filedialog.askopenfilename(filetypes=filetypes)
    if not input_file:
        return

    input_file_var.set(input_file)
    settings = collect_gui_settings()

    # 새로운 통합 처리 함수 호출
    start_processing_with_settings(input_file, settings)


def generate_srt_only_callback():
    """SRT만 생성하는 콜백"""
    # 플랫폼별 파일타입 형식 분기처리
    if IS_WINDOWS:
        filetypes = [
            ('Audio Files', '*.wav;*.mp3'),
            ('WAV files', '*.wav'),
            ('MP3 files', '*.mp3'),
            ('All files', '*.*')
        ]
    else:
        filetypes = [
            ('Audio Files', '*.wav *.mp3'),
            ('WAV files', '*.wav'),
            ('MP3 files', '*.mp3'),
            ('All files', '*.*')
        ]

    input_file = filedialog.askopenfilename(filetypes=filetypes)
    if not input_file:
        return

    input_file_var.set(input_file)
    srt_worker = generate_srt_only(input_file)
    threading.Thread(target=srt_worker, daemon=True).start()


def merge_custom_callback():
    """커스텀 병합 콜백"""
    try:
        # load original segments and duration
        base = os.path.splitext(os.path.basename(input_file_var.get()))[0]
        input_ext = os.path.splitext(input_file_var.get())[1]
        out_dir = os.path.join(os.getcwd(), 'split_audio', base)
        srt_path = os.path.join(out_dir, f"{base}{input_ext}.srt")
        segments = parse_srt_segments(srt_path)
        # load original full audio directly from the input path
        input_path = input_file_var.get()
        orig_audio = AudioSegment.from_file(input_path)
        orig_dur = len(orig_audio)
        # parse user timings list
        import json
        timings = json.loads(merge_entry.get())
        # merge using only specified segments indices
        selected = [segments[i - 1] for i in timings]

        # 사용자가 선택한 합성 타입에 따라 소스 폴더 결정
        synthesis_type = synthesis_type_var.get()
        if synthesis_type == "Instruct2":
            source_dir = os.path.join(out_dir, 'cosy_output', 'instruct')
            merged_filename = f"{base}_instruct_custom.wav"
        else:  # Zero-shot (기본값)
            source_dir = os.path.join(out_dir, 'cosy_output')
            merged_filename = f"{base}_custom.wav"

        merged_path = os.path.join(out_dir, merged_filename)
        settings = collect_gui_settings()

        merge_segments_preserve_timing(selected, orig_dur, source_dir, merged_path,
                                       length_handling=settings['length_handling'],
                                       overlap_handling=settings['overlap_handling'],
                                       max_extension=settings['max_extension'],
                                       enable_smart_compression=settings['enable_smart_compression'])
        from utils import log_message
        log_message(f"✅ {synthesis_type} 커스텀 병합 완료: {merged_filename}")
    except Exception as e:
        from utils import log_message
        log_message(f"병합 오류: {e}")


def merge_all_segments_callback():
    """전체 세그먼트 병합 콜백"""
    try:
        base = os.path.splitext(os.path.basename(input_file_var.get()))[0]
        input_ext = os.path.splitext(input_file_var.get())[1]
        out_dir = os.path.join(os.getcwd(), 'split_audio', base)
        srt_path = os.path.join(out_dir, f"{base}{input_ext}.srt")
        segments = parse_srt_segments(srt_path)
        input_path = input_file_var.get()
        orig_audio = AudioSegment.from_file(input_path)
        original_duration_ms = len(orig_audio)

        # 사용자가 선택한 합성 타입에 따라 소스 폴더 결정
        synthesis_type = synthesis_type_var.get()
        if synthesis_type == "Instruct2":
            source_dir = os.path.join(out_dir, 'cosy_output', 'instruct')
            merged_filename = f"{base}_instruct_merged.wav"
        else:  # Zero-shot (기본값)
            source_dir = os.path.join(out_dir, 'cosy_output')
            merged_filename = f"{base}_cosy_merged.wav"

        merged_path = os.path.join(out_dir, merged_filename)
        settings = collect_gui_settings()

        merge_segments_preserve_timing(segments, original_duration_ms, source_dir, merged_path,
                                       length_handling=settings['length_handling'],
                                       overlap_handling=settings['overlap_handling'],
                                       max_extension=settings['max_extension'],
                                       enable_smart_compression=settings['enable_smart_compression'])
        from utils import log_message
        log_message(f"✅ {synthesis_type} 결과 병합 완료: {merged_filename}")
    except Exception as e:
        from utils import log_message
        log_message(f"전체 병합 오류: {e}")


def save_config():
    """VAD 설정 저장"""
    try:
        vad_config = {}
        vad_config['threshold'] = float(threshold_entry.get())
        vad_config['min_speech_duration_ms'] = int(min_speech_entry.get())
        vad_config['max_speech_duration_s'] = float(max_speech_entry.get())
        vad_config['min_silence_duration_ms'] = int(min_silence_entry.get())
        vad_config['speech_pad_ms'] = int(pad_entry.get())

        success = save_vad_config(vad_config)
        if success:
            from utils import log_message
            log_message('설정 저장 완료')
        else:
            from utils import log_message
            log_message('설정 저장 실패')
    except Exception as e:
        from utils import log_message
        log_message(f'설정 저장 오류: {e}')


def set_preset_command(cmd):
    """사전 정의된 명령어 설정"""
    manual_command_var.set(cmd)


def speaker_diarization_callback():
    """화자 분리 콜백"""
    try:
        input_file = input_file_var.get()
        if not input_file:
            from utils import log_message
            log_message('먼저 파일을 선택해주세요')
            return

        settings = collect_gui_settings()
        test_speaker_diarization(input_file, settings)
        from utils import log_message
        log_message('화자 분리 완료')
    except Exception as e:
        from utils import log_message
        log_message(f'화자 분리 오류: {e}')


# GUI 구성
def create_gui():
    global root, input_file_var, log_text
    global enable_instruct_var, command_mode_var, manual_command_var
    global synthesis_type_var, length_handling_var, overlap_handling_var, max_extension_var
    global enable_english_var, enable_chinese_var, enable_japanese_var
    global translation_length_var, translation_quality_var, enable_smart_compression_var
    global enable_speaker_diarization_var, speaker_mode_var, num_speakers_var
    global enable_speaker_splitting_var, enable_3sec_extension_var
    global merge_entry
    global threshold_entry, min_speech_entry, max_speech_entry, min_silence_entry, pad_entry

    root = tk.Tk()
    input_file_var = tk.StringVar()
    root.title('STT Voice Splitter - Video & Audio Processing')
    root.geometry('1400x900')

    nb = ttk.Notebook(root)
    main_tab = ttk.Frame(nb)
    settings_tab = ttk.Frame(nb)
    log_tab = ttk.Frame(nb)
    nb.add(main_tab, text='메인')
    nb.add(settings_tab, text='설정')
    nb.add(log_tab, text='로그')
    nb.pack(expand=1, fill='both')

    # 메인 처리 버튼들
    ttk.Button(main_tab, text='🎬 영상/음성 처리 시작', command=start_processing).pack(pady=5)
    ttk.Button(main_tab, text='📝 SRT 전용 생성', command=generate_srt_only_callback).pack(pady=5)
    ttk.Button(main_tab, text='🗣️ 화자 분리 실행', command=speaker_diarization_callback).pack(pady=5)

    # 합성 타입 선택
    synthesis_frame = ttk.Frame(main_tab)
    synthesis_frame.pack(pady=5)

    # Instruct2 활성화 체크박스
    enable_instruct_var = tk.BooleanVar(value=False)
    instruct_checkbox = ttk.Checkbutton(synthesis_frame, text='Instruct2 합성 활성화',
                                        variable=enable_instruct_var)
    instruct_checkbox.pack(side=tk.LEFT, padx=(0, 20))

    # 병합할 합성 결과 선택
    ttk.Label(synthesis_frame, text='병합할 합성 결과:').pack(side=tk.LEFT, padx=(0, 10))
    synthesis_type_var = tk.StringVar(value="Zero-shot")
    ttk.Radiobutton(synthesis_frame, text='Zero-shot', variable=synthesis_type_var, value='Zero-shot').pack(
        side=tk.LEFT, padx=5)
    ttk.Radiobutton(synthesis_frame, text='Instruct2', variable=synthesis_type_var, value='Instruct2').pack(
        side=tk.LEFT, padx=5)

    # Instruct2 설정 프레임
    instruct_frame = ttk.LabelFrame(main_tab, text="Instruct2 설정")
    instruct_frame.pack(pady=5, padx=10, fill='x')

    # 명령어 입력 방식 선택
    command_mode_var = tk.StringVar(value="auto")
    ttk.Label(instruct_frame, text="명령어 설정:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
    ttk.Radiobutton(instruct_frame, text='자동 분석', variable=command_mode_var, value='auto').grid(row=0, column=1,
                                                                                                sticky='w', padx=5)
    ttk.Radiobutton(instruct_frame, text='수동 입력', variable=command_mode_var, value='manual').grid(row=0, column=2,
                                                                                                  sticky='w', padx=5)

    # 수동 명령어 입력
    manual_command_var = tk.StringVar(value="자연스럽게 말해")
    ttk.Label(instruct_frame, text="명령어:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
    manual_command_entry = ttk.Entry(instruct_frame, textvariable=manual_command_var, width=40)
    manual_command_entry.grid(row=1, column=1, columnspan=2, sticky='w', padx=5, pady=2)

    # 사전 정의된 명령어 버튼들
    preset_frame = ttk.Frame(instruct_frame)
    preset_frame.grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=5)

    preset_commands = [
        "자연스럽게 말해", "활기차게 말해", "차분하게 말해",
        "감정적으로 말해", "천천히 말해", "빠르게 말해"
    ]

    for i, cmd in enumerate(preset_commands):
        btn = ttk.Button(preset_frame, text=cmd,
                         command=lambda c=cmd: set_preset_command(c))
        btn.grid(row=i // 3, column=i % 3, padx=2, pady=2, sticky='w')

    # 타임라인 및 길이 처리 설정
    timeline_frame = ttk.LabelFrame(main_tab, text="타임라인 & 길이 처리")
    timeline_frame.pack(pady=5, padx=10, fill='x')

    length_handling_var = tk.StringVar(value="preserve")
    ttk.Label(timeline_frame, text="합성 음성 길이 처리:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
    ttk.Radiobutton(timeline_frame, text='완전 보존 (추천)', variable=length_handling_var, value='preserve').grid(row=0,
                                                                                                            column=1,
                                                                                                            sticky='w',
                                                                                                            padx=5)
    ttk.Radiobutton(timeline_frame, text='원본 길이 맞춤', variable=length_handling_var, value='fit').grid(row=0, column=2,
                                                                                                     sticky='w', padx=5)

    overlap_handling_var = tk.StringVar(value="fade")
    ttk.Label(timeline_frame, text="세그먼트 겹침 처리:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
    ttk.Radiobutton(timeline_frame, text='페이드 처리 (추천)', variable=overlap_handling_var, value='fade').grid(row=1,
                                                                                                          column=1,
                                                                                                          sticky='w',
                                                                                                          padx=5)
    ttk.Radiobutton(timeline_frame, text='자르기', variable=overlap_handling_var, value='cut').grid(row=1, column=2,
                                                                                                 sticky='w', padx=5)

    max_extension_var = tk.StringVar(value="50")
    ttk.Label(timeline_frame, text="최대 확장율 (%):").grid(row=2, column=0, sticky='w', padx=5, pady=2)
    max_extension_entry = ttk.Entry(timeline_frame, textvariable=max_extension_var, width=10)
    max_extension_entry.grid(row=2, column=1, sticky='w', padx=5, pady=2)

    # 번역 설정 프레임
    translation_frame = ttk.LabelFrame(main_tab, text="다국어 번역 설정")
    translation_frame.pack(pady=5, padx=10, fill='x')

    # 번역할 언어 선택
    target_languages_frame = ttk.Frame(translation_frame)
    target_languages_frame.grid(row=0, column=0, columnspan=4, sticky='w', padx=5, pady=5)

    ttk.Label(target_languages_frame, text="번역할 언어:").pack(side=tk.LEFT, padx=(0, 10))

    enable_english_var = tk.BooleanVar(value=True)
    enable_chinese_var = tk.BooleanVar(value=True)
    enable_japanese_var = tk.BooleanVar(value=True)

    ttk.Checkbutton(target_languages_frame, text="🇺🇸 영어", variable=enable_english_var).pack(side=tk.LEFT, padx=5)
    ttk.Checkbutton(target_languages_frame, text="🇨🇳 중국어", variable=enable_chinese_var).pack(side=tk.LEFT, padx=5)
    ttk.Checkbutton(target_languages_frame, text="🇯🇵 일본어", variable=enable_japanese_var).pack(side=tk.LEFT, padx=5)

    # 번역 길이 비율
    translation_length_var = tk.StringVar(value="1.0")
    ttk.Label(translation_frame, text="번역 길이 비율:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
    translation_length_entry = ttk.Entry(translation_frame, textvariable=translation_length_var, width=10)
    translation_length_entry.grid(row=1, column=1, sticky='w', padx=5, pady=2)

    # 번역 품질 설정
    translation_quality_var = tk.StringVar(value="accurate")
    ttk.Label(translation_frame, text="번역 품질 우선순위:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
    ttk.Radiobutton(translation_frame, text='간결함 우선', variable=translation_quality_var, value='concise').grid(row=2,
                                                                                                              column=1,
                                                                                                              sticky='w',
                                                                                                              padx=5)
    ttk.Radiobutton(translation_frame, text='균형', variable=translation_quality_var, value='balanced').grid(row=2,
                                                                                                           column=2,
                                                                                                           sticky='w',
                                                                                                           padx=5)
    ttk.Radiobutton(translation_frame, text='정확성 우선', variable=translation_quality_var, value='accurate').grid(row=2,
                                                                                                               column=3,
                                                                                                               sticky='w',
                                                                                                               padx=5)

    # 스마트 압축 설정
    smart_compression_frame = ttk.LabelFrame(main_tab, text="스마트 압축 (AI 길이 조절)")
    smart_compression_frame.pack(pady=5, padx=10, fill='x')

    enable_smart_compression_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(smart_compression_frame, text='스마트 압축 활성화 (AI 기반 자동 길이 조절)',
                    variable=enable_smart_compression_var).grid(row=0, column=0, columnspan=3, sticky='w', padx=5,
                                                                pady=2)

    # 화자 분리 설정
    speaker_diarization_frame = ttk.LabelFrame(main_tab, text="화자 분리 설정")
    speaker_diarization_frame.pack(pady=5, padx=10, fill='x')

    enable_speaker_diarization_var = tk.BooleanVar(value=False)
    speaker_mode_var = tk.StringVar(value="auto")
    num_speakers_var = tk.StringVar(value="2")
    enable_speaker_splitting_var = tk.BooleanVar(value=False)
    enable_3sec_extension_var = tk.BooleanVar(value=True)

    ttk.Checkbutton(speaker_diarization_frame, text='화자 분리 활성화', variable=enable_speaker_diarization_var).grid(row=0,
                                                                                                               column=0,
                                                                                                               sticky='w',
                                                                                                               padx=5,
                                                                                                               pady=2)
    ttk.Label(speaker_diarization_frame, text="화자 수 설정:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
    ttk.Radiobutton(speaker_diarization_frame, text='자동 감지', variable=speaker_mode_var, value='auto').grid(row=1,
                                                                                                           column=1,
                                                                                                           sticky='w',
                                                                                                           padx=5)
    ttk.Radiobutton(speaker_diarization_frame, text='고정 수', variable=speaker_mode_var, value='fixed').grid(row=1,
                                                                                                           column=2,
                                                                                                           sticky='w',
                                                                                                           padx=5)

    num_speakers_entry = ttk.Entry(speaker_diarization_frame, textvariable=num_speakers_var, width=10)
    num_speakers_entry.grid(row=1, column=3, sticky='w', padx=5, pady=2)

    # 화자 기반 분할 활성화 체크박스
    ttk.Checkbutton(speaker_diarization_frame, text='화자 기반 분할 활성화', variable=enable_speaker_splitting_var).grid(
        row=2, column=0, sticky='w', padx=5, pady=2
    )

    # 3초 확장 활성화 체크박스
    ttk.Checkbutton(speaker_diarization_frame, text='3초 확장 활성화 (제로샷 품질 개선)', variable=enable_3sec_extension_var).grid(
        row=3, column=0, sticky='w', padx=5, pady=2
    )

    # 병합 버튼 및 입력란
    ttk.Label(main_tab, text='커스텀 병합 타이밍(ms) 리스트 (JSON 배열):').pack(pady=(10, 0))
    merge_entry = ttk.Entry(main_tab, width=80)
    merge_entry.pack(pady=2)
    ttk.Button(main_tab, text='병합 실행', command=merge_custom_callback).pack(pady=5)
    ttk.Button(main_tab, text='전체 병합', command=merge_all_segments_callback).pack(pady=5)

    # 로그 탭
    log_text = scrolledtext.ScrolledText(log_tab, width=100, height=40,
                                         font=("Malgun Gothic", 10), state='disabled', background='black',
                                         foreground='lime')
    log_text.pack(padx=5, pady=5, fill='both', expand=True)

    # 설정 탭
    frm = ttk.Frame(settings_tab)
    frm.pack(padx=10, pady=10, fill='x')
    labels_k = ['음성 임계값', '최소 음성(ms)', '최대 음성(s)', '최소 무음(ms)', '음성 패딩(ms)']
    config_keys = ['threshold', 'min_speech_duration_ms', 'max_speech_duration_s', 'min_silence_duration_ms',
                   'speech_pad_ms']
    entries = []

    for i, key in enumerate(config_keys):
        ttk.Label(frm, text=labels_k[i]).grid(row=i, column=0, sticky='w')
        ent = ttk.Entry(frm)
        ent.grid(row=i, column=1, sticky='w')
        entries.append(ent)

    threshold_entry, min_speech_entry, max_speech_entry, min_silence_entry, pad_entry = entries
    ttk.Button(frm, text='저장', command=save_config).grid(row=len(entries), column=0, columnspan=2, pady=10)

    # VAD 설정 로드
    vad_config = load_vad_config()
    for ent, key in zip(entries, config_keys):
        ent.insert(0, vad_config[key])

    # 로그 큐 처리 시작
    root.after(200, process_log_queue)

    return root


# 헤드리스 모드로 실행
def run_headless():
    """헤드리스 환경에서 실행될 명령행 인터페이스 구현"""
    parser = argparse.ArgumentParser(description='헤드리스 모드로 실행')
    parser.add_argument('--input', type=str, required=True, help='입력 파일 경로')
    parser.add_argument('--headless', action='store_true', help='헤드리스 모드로 실행')

    args = parser.parse_args()

    if args.headless:
        settings = {
            'enable_instruct': False,
            'manual_command': None,
            'length_handling': "preserve",
            'overlap_handling': "fade",
            'max_extension': 50,
            'selected_languages': ['english', 'chinese', 'japanese'],
            'translation_length': 1.0,
            'quality_mode': "accurate",
            'enable_smart_compression': True,
            'enable_speaker_diarization': False,
            'num_speakers': None,
            'enable_speaker_splitting': False,
            'enable_3sec_extension': True,
            'vocals_volume': 1.0,
            'background_volume': 0.8
        }

        start_processing_with_settings(args.input, settings)
        return

    # GUI 모드 실행
    root = create_gui()
    root.mainloop()


if __name__ == "__main__":
    # GUI 사용 불가능 환경 감지
    try:
        # tkinter GUI 초기화 시도
        test_root = tk.Tk()
        test_root.withdraw()  # 숨기기
        test_root.destroy()   # 정리

        # GUI 사용 가능하면 정상 실행
        root = create_gui()
        root.mainloop()

    except (tk.TclError, ImportError):
        # GUI 사용 불가능하면 헤드리스 모드 안내
        print("GUI를 사용할 수 없는 환경입니다.")
        print("헤드리스 모드로 실행하려면:")
        print("python STT_Voice_Spliter_NEW.py --headless --input [파일경로]")

        # 명령행 인자가 있다면 헤드리스 모드 실행
        if len(sys.argv) > 1:
            run_headless()
        else:
            sys.exit(1)
