# Load environment variables from .env file
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $ScriptDir ".env"

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $key, $value = $line -split "=", 2
            [Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim(), "Process")
        }
    }
}

# Check if prompt is provided
if ($args.Count -eq 0) {
    Write-Host "Usage: .\chat_completion.ps1 `"your prompt here`""
    exit 1
}

$Prompt = $args[0]

# OpenAI API endpoint
$Endpoint = if ($env:OPENAI_ENDPOINT) { $env:OPENAI_ENDPOINT } else { "https://api.openai.com/v1/chat/completions" }
$Model = if ($env:OPENAI_MODEL) { $env:OPENAI_MODEL } else { "gpt-4.1" }
$Seed = if ($env:OPENAI_SEED) { [int]$env:OPENAI_SEED } else { 42 }
$Temperature = if ($env:OPENAI_TEMPERATURE) { [double]$env:OPENAI_TEMPERATURE } else { 0 }

# Call the API with deterministic settings:
#   seed        - fixes the random seed for reproducible outputs
#   temperature - 0 means no randomness
#   top_p       - 1 is default
$Body = @{
    model    = $Model
    messages = @(
        @{ role = "system"; content = "You are a helpful assistant." }
        @{ role = "user";   content = $Prompt }
    )
    seed                  = $Seed
    temperature           = $Temperature
    top_p                 = 1
    max_completion_tokens = 2048
} | ConvertTo-Json -Depth 4

$Response = Invoke-RestMethod -Uri $Endpoint `
    -Method Post `
    -ContentType "application/json" `
    -Headers @{ "Authorization" = "Bearer $($env:OPENAI_API_KEY)" } `
    -Body $Body

$Response | ConvertTo-Json -Depth 10
