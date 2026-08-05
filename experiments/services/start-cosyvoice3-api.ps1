# Start the CosyVoice3 API required by CosyVoice TTS experiments. Keep this window open.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$python = "D:\anaconda\envs\cosyvoice\python.exe"
$server = Join-Path $root "third_party\CosyVoice\runtime\python\fastapi\server.py"
$model = "D:\models\Fun-CosyVoice3-0.5B-2512"
$port = 50000

if (-not (Test-Path -LiteralPath $python)) { throw "CosyVoice Python not found: $python" }
if (-not (Test-Path -LiteralPath $server)) { throw "CosyVoice FastAPI server not found: $server" }
if (-not (Test-Path -LiteralPath $model)) { throw "CosyVoice model not found: $model" }

Write-Host "Starting CosyVoice3 API on http://127.0.0.1:$port"
& $python $server --port $port --model_dir $model
exit $LASTEXITCODE
