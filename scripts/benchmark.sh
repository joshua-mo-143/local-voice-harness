#!/usr/bin/env bash
set -eo pipefail

VH="/home/joshuam/local-voice-harness/.venv/bin/voice-harness"
PY="/home/joshuam/local-voice-harness/.venv/bin/python"
WAV="/run/user/1000/voice-harness/request.wav"
LLM_CHAT="http://127.0.0.1:8090/v1/chat/completions"

bench_llm() {
  local label="$1" prompt="$2" max_tokens="${3:-128}"
  local payload
  payload="$("$PY" -c "
import json, sys
print(json.dumps({
    'model': 'qwen3.5-9b',
    'messages': [{'role': 'user', 'content': sys.argv[1]}],
    'temperature': 0.7,
    'max_tokens': int(sys.argv[2]),
    'stream': False,
}))" "$prompt" "$max_tokens")"
  local t0 t1 elapsed curl_time content
  t0=$(date +%s.%N)
  curl -s -o /tmp/llm_bench.json \
    -w "%{time_total}" \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "$LLM_CHAT" > /tmp/llm_curl_time
  t1=$(date +%s.%N)
  elapsed=$(awk "BEGIN {printf \"%.3f\", $t1 - $t0}")
  curl_time=$(cat /tmp/llm_curl_time)
  content="$("$PY" -c "import json; d=json.load(open('/tmp/llm_bench.json')); print(d['choices'][0]['message'].get('content','')[:120])" 2>/dev/null || echo "(parse error)")"
  echo "  ${label}: wall=${elapsed}s curl=${curl_time}s"
  echo "  Response: ${content}"
}

wait_llm() {
  for _ in $(seq 1 120); do
    curl -sf http://127.0.0.1:8090/health >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  return 1
}

wait_tts() {
  for _ in $(seq 1 120); do
    if "$VH" status 2>/dev/null | grep -q '"tts_ready": true'; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

echo "=== Voice Harness Performance Benchmark ==="
echo "Date: $(date -Iseconds)"
echo ""
echo "--- Hardware ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "Qwen3.5-9B-UD-Q4_K_XL | Whisper large-v3-turbo | Chatterbox Turbo"
echo ""

systemctl --user stop voice-harness-wake.service dictation.service voice-harness-llm.service voice-harness-tts.service 2>/dev/null || true
sleep 3
rm -f "${XDG_RUNTIME_DIR}/voice-harness-tts.sock"
echo "GPU idle: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo ""

echo "--- STT (Whisper large-v3-turbo, CUDA) ---"
systemctl --user start dictation.service
sleep 2
for _ in $(seq 1 30); do
  if "$VH" status 2>/dev/null | grep -q '"stt_ready": true'; then
    break
  fi
  sleep 1
done
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WAV" 2>/dev/null || echo 0)
echo "Audio: ${DUR}s ($(stat -c%s "$WAV") bytes)"
for run in 1 2 3; do
  T0=$(date +%s.%N)
  OUT=$("$PY" -c "
from local_voice_harness.stt.client import transcribe
from local_voice_harness.config import WAV_PATH
print(transcribe(WAV_PATH))
" 2>&1)
  T1=$(date +%s.%N)
  E2E=$(awk "BEGIN {printf \"%.3f\", $T1 - $T0}")
  STT_LINE=$(echo "$OUT" | grep '"stage": "stt"' || true)
  TEXT=$(echo "$OUT" | grep -v '^{' | tail -1)
  echo "Run ${run}: ${STT_LINE} | e2e=${E2E}s | text=\"${TEXT}\""
done
echo "GPU with STT: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo ""

echo "--- LLM (Qwen3.5-9B, CUDA) ---"
systemctl --user stop dictation.service
sleep 2
T0=$(date +%s.%N)
systemctl --user start voice-harness-llm.service
wait_llm
T1=$(date +%s.%N)
LOAD=$(awk "BEGIN {printf \"%.3f\", $T1 - $T0}")
echo "Cold load: ${LOAD}s"
echo "GPU with LLM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "Warm inference:"
bench_llm "short" "What is two plus two?" 64
bench_llm "medium" "Why is the sky blue?" 128
echo ""

echo "--- TTS (Chatterbox Turbo, CUDA) ---"
systemctl --user stop voice-harness-llm.service
sleep 3
T0=$(date +%s.%N)
systemctl --user start voice-harness-tts.service
wait_tts
T1=$(date +%s.%N)
LOAD=$(awk "BEGIN {printf \"%.3f\", $T1 - $T0}")
echo "Cold load: ${LOAD}s"
echo "GPU with TTS: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
"$PY" - <<'PY'
import json
import time

from local_voice_harness.tts.client import stream_and_play

cases = [
    "Hello there.",
    "The sky is blue because sunlight scatters in the atmosphere.",
    "Computing began with mechanical calculators. Early electronic computers used vacuum tubes. Modern chips contain billions of transistors.",
]
for index, text in enumerate(cases, 1):
    started = time.perf_counter()
    result = stream_and_play(text)
    elapsed = round(time.perf_counter() - started, 3)
    summary = {
        key: result[key]
        for key in (
            "generation_seconds",
            "audio_seconds",
            "realtime_factor",
            "chunks",
            "request_seconds",
        )
        if key in result
    }
    print(f"Run {index}: {json.dumps(summary)} | wall={elapsed}s")
PY
echo ""

echo "--- Full Pipeline (LLM + TTS warm, post-transcription path) ---"
systemctl --user start voice-harness-llm.service
wait_llm
echo "GPU with LLM+TTS: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
QUERIES=(
  "What is two plus two?"
  "Why is the sky blue?"
  "Tell me a fun fact about octopuses."
)
for index in "${!QUERIES[@]}"; do
  query="${QUERIES[$index]}"
  run=$((index + 1))
  echo "Run ${run}: \"${query}\""
  T0=$(date +%s.%N)
  OUT=$("$VH" text "$query" 2>&1)
  T1=$(date +%s.%N)
  E2E=$(awk "BEGIN {printf \"%.3f\", $T1 - $T0}")
  echo "$OUT" | grep -E '^\{|^Assistant:'
  echo "{\"stage\": \"e2e_pipeline\", \"run\": ${run}, \"seconds\": ${E2E}}"
  echo ""
done

echo "--- VRAM Coexistence (STT + LLM + TTS) ---"
systemctl --user start dictation.service
sleep 2
echo "All three services started."
echo "GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
"$VH" status
echo ""

systemctl --user stop voice-harness-llm.service voice-harness-tts.service 2>/dev/null || true
systemctl --user start dictation.service voice-harness-wake.service
echo "Services restored (wake + dictation)."
