$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
$katalaTemp = $null
try {
    $katalaManifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'MANIFEST.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $katalaOutput = Join-Path $PSScriptRoot $katalaManifest.file.name
    if (Test-Path -LiteralPath $katalaOutput) {
        if ((Get-FileHash -LiteralPath $katalaOutput -Algorithm SHA256).Hash -eq $katalaManifest.file.sha256) {
            Write-Host 'Файл уже собран и проверен:' -ForegroundColor Green
            Write-Host $katalaOutput
            exit 0
        }
        throw 'Файл с таким именем уже есть. Переместите его перед сборкой.'
    }
    foreach ($katalaPart in $katalaManifest.parts) {
        $katalaPath = Join-Path $PSScriptRoot $katalaPart.path
        if (-not (Test-Path -LiteralPath $katalaPath -PathType Leaf)) {
            throw ('Не найдена часть: ' + $katalaPart.path + '. Распакуйте весь архив, сохраняя папки.')
        }
        if ((Get-Item -LiteralPath $katalaPath).Length -ne $katalaPart.size) {
            throw ('Неверный размер части: ' + $katalaPart.path)
        }
        if ((Get-FileHash -LiteralPath $katalaPath -Algorithm SHA256).Hash -ne $katalaPart.sha256) {
            throw ('Повреждена часть: ' + $katalaPart.path + '. Скачайте архив заново.')
        }
    }
    $katalaTemp = Join-Path $PSScriptRoot ('Katala_' + [guid]::NewGuid().ToString('N') + '.assembling')
    $katalaStream = [IO.File]::Open($katalaTemp, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try {
        $katalaIndex = 0
        foreach ($katalaPart in $katalaManifest.parts) {
            $katalaIndex++
            Write-Host ('Сборка: часть {0} из {1}...' -f $katalaIndex, $katalaManifest.parts.Count)
            $katalaInput = [IO.File]::OpenRead((Join-Path $PSScriptRoot $katalaPart.path))
            try { $katalaInput.CopyTo($katalaStream) }
            finally { $katalaInput.Dispose() }
        }
    }
    finally { $katalaStream.Dispose() }
    Write-Host 'Проверка готового видео...'
    if ((Get-Item -LiteralPath $katalaTemp).Length -ne $katalaManifest.file.size) {
        throw 'Размер готового файла не совпал с ожидаемым.'
    }
    if ((Get-FileHash -LiteralPath $katalaTemp -Algorithm SHA256).Hash -ne $katalaManifest.file.sha256) {
        throw 'Контрольная сумма готового файла не совпала.'
    }
    [IO.File]::Move($katalaTemp, $katalaOutput)
    $katalaTemp = $null
    Write-Host 'ГОТОВО. Видео собрано, целостность подтверждена.' -ForegroundColor Green
    Write-Host $katalaOutput
}
catch {
    Write-Host ('ОШИБКА: ' + $_.Exception.Message) -ForegroundColor Red
    if ($null -ne $katalaTemp -and (Test-Path -LiteralPath $katalaTemp)) {
        Remove-Item -LiteralPath $katalaTemp
    }
    exit 1
}
