#!/usr/bin/env bash
# Run the five demonstration conversations and save answer + raw tool evidence to evidence/.
# Usage: scripts/run_conversations.sh [1-5]   (no argument = all)
set -uo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
PY=.venv/bin/python

declare -A Q=(
  [1]="Which dataset version was used to train the best run of the mlops-agent-training experiment, and how do I reproduce exactly that dataset?"
  [2]="Compare the last five training runs in mlops-agent-training and recommend the best candidate based on F1 score."
  [3]="Trigger the model training pipeline and report whether execution succeeded."
  [4]="Why is my prediction API unavailable?"
  [5]="Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
)
NAMES=([1]=dataset_provenance [2]=model_selection [3]=pipeline_execution [4]=deployment_diagnosis [5]=multi_tool_reasoning)

for i in ${1:-1 2 3 4 5}; do
  echo; echo "################ Conversation $i: ${NAMES[$i]}"
  echo "you> ${Q[$i]}"
  $PY -W ignore -m agent.main "${Q[$i]}" --save "evidence/conversation_${i}_${NAMES[$i]}.json"
done
