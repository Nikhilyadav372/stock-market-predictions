param(
    [Parameter(Mandatory=$true)]
    [string]$HfUsername,
    
    [Parameter(Mandatory=$true)]
    [string]$HfToken,
    
    [string]$SpaceName = "stock-market-api"
)

$ErrorActionPreference = "Stop"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "🚀 Hugging Face Space Deployment Automation" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

$tempDir = Join-Path $env:TEMP "hf-stock-deploy-$(Get-Random)"
$repoUrl = "https://${HfUsername}:${HfToken}@huggingface.co/spaces/${HfUsername}/${SpaceName}"
$sourceBackend = "C:\Users\nikhil\Desktop\stock market project\backend"

Write-Host "📁 Creating temporary directory: $tempDir" -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

try {
    Write-Host "📥 Cloning Hugging Face Space: ${HfUsername}/${SpaceName}..." -ForegroundColor Yellow
    git clone $repoUrl $tempDir

    Write-Host "📋 Copying backend files..." -ForegroundColor Yellow
    # Copy app directory
    Copy-Item -Path "$sourceBackend\app" -Destination "$tempDir\app" -Recurse -Force
    # Remove any __pycache__ inside app
    Get-ChildItem -Path "$tempDir\app" -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    # Copy requirements & config files
    Copy-Item -Path "$sourceBackend\requirements.txt" -Destination "$tempDir\requirements.txt" -Force

    # Ensure directories exist
    New-Item -ItemType Directory -Force -Path "$tempDir\models" | Out-Null
    New-Item -ItemType Directory -Force -Path "$tempDir\data\raw" | Out-Null
    New-Item -ItemType Directory -Force -Path "$tempDir\data\processed" | Out-Null
    New-Item -ItemType Directory -Force -Path "$tempDir\data\sample" | Out-Null

    # Create Dockerfile with port 7860
    $dockerfileContent = @"
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/models /app/data/raw /app/data/processed /app/data/sample

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
"@
    Set-Content -Path "$tempDir\Dockerfile" -Value $dockerfileContent -Encoding UTF8

    # Create README.md with HF metadata
    $readmeContent = @"
---
title: Stock Market API
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Stock Market Forecasting API
FastAPI backend with ML & PyTorch models.
"@
    Set-Content -Path "$tempDir\README.md" -Value $readmeContent -Encoding UTF8

    # Create .gitignore
    $gitignoreContent = @"
__pycache__/
*.pyc
*.pyo
.pytest_cache/
venv/
.env
data/raw/*
data/processed/*
models/*.pkl
models/*.pt
"@
    Set-Content -Path "$tempDir\.gitignore" -Value $gitignoreContent -Encoding UTF8

    # Commit and push
    Push-Location $tempDir
    Write-Host "📦 Staging files for Git..." -ForegroundColor Yellow
    git config user.name "Nikhil Yadav"
    git config user.email "nikhil@users.noreply.huggingface.co"
    git add .
    git commit -m "Deploy stock forecasting backend with Docker" --allow-empty
    Write-Host "🚀 Pushing to Hugging Face Spaces..." -ForegroundColor Green
    git push origin main

    Pop-Location

    Write-Host "`n🎉 SUCCESS! Your backend is deploying on Hugging Face Spaces!" -ForegroundColor Green
    Write-Host "🌐 Space URL: https://huggingface.co/spaces/${HfUsername}/${SpaceName}" -ForegroundColor Cyan
    Write-Host "🔗 API URL:   https://${HfUsername}-${SpaceName}.hf.space" -ForegroundColor Cyan
    Write-Host "📖 API Docs:  https://${HfUsername}-${SpaceName}.hf.space/docs`n" -ForegroundColor Cyan
}
catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}
finally {
    if (Test-Path $tempDir) {
        Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
    }
}
