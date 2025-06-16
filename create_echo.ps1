<#
  create_echo_env.ps1
  ����������������������������������������������������������������������������������������������������������������������������
  Conda + Poetry + PyTorch(GPU) + EchoInStone �ڵ� ���� ��ũ��Ʈ
  Tested on: Windows 11, CUDA Driver 12.9, Anaconda/Miniconda
  Usage   : �� Anaconda Prompt(PowerShell)���� ����
            �� PS> .\create_echo_env.ps1 [-�ɼ�]
  �ɼ� �׸�:
    -EnvName   <string>  �� conda ȯ�� �̸�        (default: echo_env)
    -PythonVer <string>  Python ����               (default: 3.11)
    -TorchVer  <string>  PyTorch ����              (default: 2.7.0)
    -CudaTag   <string>  Wheel CUDA �±� (cu128��)  (default: cu128)
#>

param(
    [string]$EnvName   = "echo_env",
    [string]$PythonVer = "3.11",
    [string]$TorchVer  = "2.7.0",
    [string]$CudaTag   = "cu128"
)

function Exec($cmd) {
    Write-Host $cmd -ForegroundColor Cyan
    & cmd /c $cmd              # exit code ����
    if ($LASTEXITCODE -ne 0) { throw "?  ����: $cmd" }
}

Write-Host "�������������������� EchoInStone ȯ�� ���� ���� ��������������������" -ForegroundColor Yellow

# 1. Conda ȯ�� ����
Exec "conda create -y -n $EnvName python=$PythonVer git ffmpeg -c conda-forge"

# 2. ȯ�� Ȱ��ȭ (conda init powershell ���� ���� �ʿ�)
Exec "conda activate $EnvName"

# 3. pip �ֽ�ȭ
Exec "python -m pip install --upgrade pip"

# 4. Poetry ��ġ
Exec "pip install poetry==1.8.3"

# 5. PyTorch GPU wheel (�� �� �������)  �� CUDA 12.8 wheel, ����̹� 12.9 ȣȯ
Exec "pip install torch==$TorchVer+$CudaTag torchvision==0.22.0+$CudaTag torchaudio==$TorchVer+$CudaTag --index-url https://download.pytorch.org/whl/$CudaTag"

# 6. EchoInStone �ҽ� Ŭ��
Exec "git clone https://github.com/jeanjerome/EchoInStone.git"
Set-Location .\EchoInStone

# 7. Poetry ������ ��ġ (����ȯ�� �ߺ� ����)
Exec "poetry config virtualenvs.create false --local"
Exec "poetry install --no-root --no-interaction"

Write-Host "`n?  ��� ��ġ �Ϸ�!" -ForegroundColor Green
Write-Host "   Conda ȯ�� ������ :  conda activate $EnvName"
Write-Host "   ���� ����        :  poetry run python main.py `"C:\audio\sample.wav`""
Write-Host "������������������������������������������������������������������������������������������������������������������������"