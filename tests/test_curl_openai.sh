#!/usr/bin/env bash
# Smoke-test the configured OpenAI endpoint with curl.
# Uses OPENAI_KEY from env; OPENAI_TARGET defaults to https://api.openai.com.

.  .env
set -euo pipefail

TARGET=${OPENAI_TARGET:-"https://api.openai.com"}
KEY=${OPENAI_KEY:-${OPENAI_API_KEY:-}}
MODEL=${OPENAI_MODEL:-"gpt-4o"}

if [[ -z "$KEY" ]]; then
  echo "OPENAI_KEY (or OPENAI_API_KEY) is required" >&2
  exit 1
fi

echo "Testing endpoint: ${TARGET}/v1/chat/completions with ${KEY}" >&2

curl -v -sS \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${KEY}" \
  -X POST "${TARGET%/}/v1/chat/completions" \
  -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"ping\"}]}"

