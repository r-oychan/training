#!/bin/bash

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.env"

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
  echo "Error: OPENAI_API_KEY is not set in .env"
  exit 1
fi

# Check if prompt is provided
if [ -z "$1" ]; then
  echo "Usage: ./chat_completion.sh \"your prompt here\""
  exit 1
fi

PROMPT="$1"

# OpenAI API endpoint
ENDPOINT="${OPENAI_ENDPOINT:-https://api.openai.com/v1/chat/completions}"
MODEL="${OPENAI_MODEL:-gpt-4.1}"
SEED="${OPENAI_SEED:-42}"
TEMPERATURE="${OPENAI_TEMPERATURE:-0}"

# Call the API with deterministic settings:
#   seed        - fixes the random seed for reproducible outputs
#   temperature - 0 means no randomness
#   top_p       - 1 is default
curl -s "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "'"$PROMPT"'"
      }
    ],
    "seed": '"$SEED"',
    "temperature": '"$TEMPERATURE"',
    "top_p": 1,
    "max_completion_tokens": 2048
  }' | python3 -m json.tool
