# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Receive one UTF-8 JSON request on stdin and emit one UTF-8 JSON response.
# Text stays data throughout; no expression evaluation, shell interpolation, XML,
# system-volume mutation, or default-device mutation is performed.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$voice = $null
$outputs = $null

function Write-JsonResponse {
    param([object]$Value)
    [Console]::Out.WriteLine(($Value | ConvertTo-Json -Compress -Depth 6))
}

try {
    $reader = [System.IO.StreamReader]::new(
        [Console]::OpenStandardInput(), [System.Text.UTF8Encoding]::new($false, $true)
    )
    $request = ConvertFrom-Json -InputObject $reader.ReadToEnd()
    if ($null -eq $request -or $request.action -isnot [string] -or $request.action -notin @('list', 'speak')) {
        throw 'Unknown audio request action'
    }
    $voice = New-Object -ComObject SAPI.SpVoice
    $outputs = $voice.GetAudioOutputs()
    if ($request.action -eq 'list') {
        $devices = @()
        for ($index = 0; $index -lt $outputs.Count; $index++) {
            $token = $outputs.Item($index)
            $devices += @{ id = [string]$token.Id; name = [string]$token.GetDescription() }
        }
        Write-JsonResponse @{ ok = $true; outputs = @($devices) }
    }
    else {
        if ($request.text -isnot [string] -or [string]::IsNullOrWhiteSpace($request.text) -or
            $request.text.Length -gt 4000 -or $request.text.Contains([char]0)) {
            throw 'Speech text must contain 1 to 4000 characters and no NUL'
        }
        if (($request.volume -isnot [int] -and $request.volume -isnot [long]) -or
            $request.volume -lt 0 -or $request.volume -gt 100) {
            throw 'Speech volume must be an integer from 0 through 100'
        }
        if ($null -ne $request.device_id -and
            ($request.device_id -isnot [string] -or $request.device_id.Length -eq 0 -or
                $request.device_id.Contains([char]0))) {
            throw 'Audio device ID must be a nonempty string or null'
        }
        if ($null -eq $request.device_id) {
            $selected = $voice.AudioOutput
        }
        else {
            $selected = $null
            for ($index = 0; $index -lt $outputs.Count; $index++) {
                $token = $outputs.Item($index)
                if ([string]::Equals([string]$token.Id, $request.device_id, [StringComparison]::Ordinal)) {
                    $selected = $token
                    break
                }
            }
            if ($null -eq $selected) {
                throw 'The exact requested SAPI audio output is unavailable; no fallback was selected'
            }
            $voice.AudioOutput = $selected
        }
        if ($null -eq $selected) {
            throw 'SAPI has no default audio output'
        }
        $selectedInfo = @{ id = [string]$selected.Id; name = [string]$selected.GetDescription() }
        $voice.Volume = [int]$request.volume
        $requestedUtc = [DateTimeOffset]::UtcNow.ToString('o')
        # SVSFIsNotXML (16), without SVSFlagsAsync, forces synchronous plain text.
        # https://learn.microsoft.com/en-us/previous-versions/windows/desktop/ms720892(v=vs.85)
        [void]$voice.Speak($request.text, 16)
        $completedUtc = [DateTimeOffset]::UtcNow.ToString('o')
        Write-JsonResponse @{
            ok = $true
            selected_device = $selectedInfo
            volume = [int]$request.volume
            host_play_requested_utc = $requestedUtc
            host_play_completed_utc = $completedUtc
        }
    }
}
catch {
    Write-JsonResponse @{ ok = $false; error = $_.Exception.Message }
    exit 1
}
finally {
    foreach ($value in @($outputs, $voice)) {
        if ($null -ne $value -and [Runtime.InteropServices.Marshal]::IsComObject($value)) {
            try { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($value) } catch { }
        }
    }
}
