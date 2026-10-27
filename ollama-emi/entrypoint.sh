#!/bin/sh
set -e
ollama serve &
PID=$!
until ollama list > /dev/null 2>&1; do sleep 1; done
ollama show "$OLLAMA_MODEL" > /dev/null 2>&1 || ollama pull "$OLLAMA_MODEL"
wait $PID
