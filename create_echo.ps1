<#
  create_echo_env.ps1
  ──────────────────────────────────────────────────────────────
  Conda + Poetry + PyTorch(GPU) + EchoInStone 자동 구축 스크립트
  Tested on: Windows 11, CUDA Driver 12.9, Anaconda/Miniconda
  Usage   : ① Anaconda Prompt(PowerShell)에서 실행
            ② PS> .\create_echo_env.ps1 [-옵션]
  옵션 항목:
    -EnvName   <string>  새 conda 환경 이름        (default: echo_env)
    -PythonVer <string>  Python 버전               (default: 3.11)
    -TorchVer  <string>  PyTorch 버전              (default: 2.7.0)
    -CudaTag   <string>  Wheel CUDA 태그 (cu128…)  (default: cu128)
#>

param(
    [string]$EnvName   = "echo_env",
    [string]$PythonVer = "3.11",
    [string]$TorchVer  = "2.7.0",
    [string]$CudaTag   = "cu128"
)

function Exec($cmd) {
    Write-Host $cmd -ForegroundColor Cyan
    & cmd /c $cmd              # exit code 전달
    if ($LASTEXITCODE -ne 0) { throw "?  실패: $cmd" }
}

Write-Host "────────── EchoInStone 환경 구축 시작 ──────────" -ForegroundColor Yellow

# 1. Conda 환경 생성
Exec "conda create -y -n $EnvName python=$PythonVer git ffmpeg -c conda-forge"

# 2. 환경 활성화 (conda init powershell 사전 수행 필요)
Exec "conda activate $EnvName"

# 3. pip 최신화
Exec "python -m pip install --upgrade pip"

# 4. Poetry 설치
Exec "pip install poetry==1.8.3"

# 5. PyTorch GPU wheel (한 줄 명령으로)  ─ CUDA 12.8 wheel, 드라이버 12.9 호환
Exec "pip install torch==$TorchVer+$CudaTag torchvision==0.22.0+$CudaTag torchaudio==$TorchVer+$CudaTag --index-url https://download.pytorch.org/whl/$CudaTag"

# 6. EchoInStone 소스 클론
Exec "git clone https://github.com/jeanjerome/EchoInStone.git"
Set-Location .\EchoInStone

# 7. Poetry 의존성 설치 (가상환경 중복 방지)
Exec "poetry config virtualenvs.create false --local"
Exec "poetry install --no-root --no-interaction"

Write-Host "`n?  모든 설치 완료!" -ForegroundColor Green
Write-Host "   Conda 환경 재진입 :  conda activate $EnvName"
Write-Host "   예시 실행        :  poetry run python main.py `"C:\audio\sample.wav`""
Write-Host "────────────────────────────────────────────────────────────"