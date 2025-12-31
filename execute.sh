#!/bin/bash

# 1. Set response model to evaluate (uncomment or modify desired model)
RESPONSE_MODEL="gpt-4o-2024-05-13"
# RESPONSE_MODEL="claude-3-5-sonnet-20240620"

# 2. Set execution mode (whether to run in tmux session)
USE_TMUX=false

# 3. List of judge models for evaluation
MODELS=(
  "openrouter/openai/gpt-oss-20b"
  "openrouter/openai/gpt-oss-120b"
)

# Set data file path
PAIRS_FILE="data/dataset=judgebench,response_model=${RESPONSE_MODEL}.jsonl"

# Join model list with commas
JUDGE_MODELS=$(IFS=,; echo "${MODELS[*]}")

# Construct command
CMD="python run_judge.py \
  --judge_name arena_hard \
  --judge_model \"$JUDGE_MODELS\" \
  --pairs \"$PAIRS_FILE\" \
  --concurrency_limit 50"

if [ "$USE_TMUX" = true ]; then
  # tmux session name (distinguished by response model)
  SESSION_NAME="judge_${RESPONSE_MODEL//./_}"

  # Kill existing session with same name if exists
  tmux kill-session -t "$SESSION_NAME" 2>/dev/null

  # Create new tmux session and run in background
  tmux new-session -d -s "$SESSION_NAME" "$CMD"

  echo "Started tmux session: $SESSION_NAME"
  echo "Evaluating responses from: $RESPONSE_MODEL"
  echo "Using judge models:"
  for model in "${MODELS[@]}"; do
    echo "  - $model"
  done

  echo ""
  echo "Commands to manage the session:"
  echo "1. Attach to session: tmux attach -t $SESSION_NAME"
  echo "2. Detach: Ctrl + B, then D"
  echo "3. Kill session: tmux kill-session -t $SESSION_NAME"
else
  echo "Running in foreground mode..."
  echo "Evaluating responses from: $RESPONSE_MODEL"
  eval "$CMD"
fi
