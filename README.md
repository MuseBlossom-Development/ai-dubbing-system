# 오디오 파일 분할 및 자동 필사 도구

이 프로젝트는 오디오 파일(WAV 또는 MP3)을 입력 받아 음성 구간을 자동으로 분할하고, Whisper CLI를 통해 해당 구간을 텍스트로 필사하는 파이썬 기반 도구입니다.

## 주요 기능

- **오디오 파일 포맷 변환**  
  - WAV 파일인 경우 FFmpeg를 이용해 MP3로 변환  
  - 입력 파일이 이미 MP3이면 변환 없이 그대로 사용

- **오디오 분할 (Silero VAD 사용)**  
  - Silero VAD 모델을 사용해 음성 구간 감지  
  - 감지된 구간을 기준으로 MP3 파일 분할 (재인코딩 없이 `-c copy` 옵션 사용)  
  - 터미널에 진행률 바를 출력하여 작업 진행 상태를 표시

- **자동 필사 (Whisper CLI 사용)**  
  - 분할된 MP3 파일을 Whisper CLI에 전달하여 텍스트 필사  
  - 필사된 텍스트 파일의 개행(\n)을 제거 후 지정한 폴더로 이동

- **환경 및 오류 처리**  
  - FFmpeg 설치 여부를 확인하여 설치되지 않은 경우 오류 메시지를 출력하고 프로그램 종료  
  - 각 단계별 오류 발생 시 명확한 메시지 출력

## 사용 기술 및 의존성

- **프로그래밍 언어 및 라이브러리**:  
  - Python 기본 라이브러리: `sys`, `os`, `subprocess`, `shutil`, `time`  
  - PyTorch 및 torchaudio: Silero VAD 모델 로딩과 오디오 처리에 사용

- **외부 도구**:  
  - **FFmpeg**: 오디오 포맷 변환 및 분할 처리  
  - **Whisper CLI**: 자동 필사 수행  
    - 사용 모델: `models/ggml-large-v3-turbo.bin`

## 코드 구성 및 실행 순서

1. **환경 확인 및 초기 설정**  
   - `check_ffmpeg()` 함수를 통해 FFmpeg 설치 여부 확인

2. **오디오 파일 변환**  
   - `convert_to_mp3(file_path)` 함수에서 WAV 파일이면 MP3로 변환

3. **오디오 분할**  
   - `split_audio(file_path)` 함수에서  
     - 파일 이름을 기반으로 출력 폴더(`split_audio/파일명/MP3`, `split_audio/파일명/TEXT`) 생성  
     - Silero VAD 모델을 이용해 음성 구간 감지 후 MP3 파일로 분할 저장  
     - 진행률 바를 통해 분할 상태 표시

4. **텍스트 후처리 및 이동**  
   - `remove_newlines_from_text(text_file)` 함수로 필사 텍스트 파일의 개행 제거  
   - `transcribe_audio(mp3_folder, text_folder)` 함수에서 Whisper CLI를 실행해 필사 후, 생성된 TXT 파일을 지정 폴더로 이동

5. **메인 실행 흐름**  
   - 스크립트 실행 시 인자로 전달받은 오디오 파일에 대해 MP3 변환 → 오디오 분할 → 자동 필사 작업 순으로 진행

## 실행 방법

터미널에서 아래 명령어로 스크립트를 실행합니다.

```bash
python whisper_coreml_split.py [오디오 파일 경로]
