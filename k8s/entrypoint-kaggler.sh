set -e
set -o pipefail

: "${COMPETITION_DIR:?missing COMPETITION_DIR}"
: "${KAGGLER_NAME:?missing KAGGLER_NAME}"
: "${RESEARCH_TAG:?missing RESEARCH_TAG}"
: "${CLAUDE_CODE_OAUTH_TOKEN:?missing CLAUDE_CODE_OAUTH_TOKEN}"

BRANCH="$RESEARCH_TAG/kaggler/$KAGGLER_NAME"
AGENT_MODEL="${AGENT_MODEL:-}"
KAGGLER_WORKDIR="${KAGGLER_WORKDIR:-/workspace/kagent/$COMPETITION_DIR/kaggler}"
KAGGLER_PROMPT_FILE="${KAGGLER_PROMPT_FILE:-KAGGLER_AGENT.md}"
export PATH="$HOME/.claude/bin:$HOME/.local/bin:$PATH"

echo "=== kagent Kaggler: $KAGGLER_NAME ==="
echo "Competition: $COMPETITION_DIR"
echo "Model: ${AGENT_MODEL:-default}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd /workspace/kagent

# --- Branch setup (sparse: kaggler dir only, no organizer) ---
if git ls-remote --exit-code origin "refs/heads/$BRANCH" >/dev/null 2>&1; then
    git fetch origin "$BRANCH"
    git checkout -b "$BRANCH" "origin/$BRANCH"
else
    git checkout -b "$BRANCH"
    git push -u origin "$BRANCH"
fi
git sparse-checkout set "$COMPETITION_DIR/kaggler" k8s pyproject.toml .gitignore CLAUDE.md

# --- Install ---
uv pip install --system -e .
git config user.name "kagent-$KAGGLER_NAME"
git config user.email "kagent-$KAGGLER_NAME@kagent"

# Update claude to latest (image has it baked in)
claude update >/dev/null 2>&1 || echo "Claude update failed, using baked version"

# --- Register Weave Claude Code Plugin (tools baked into image) ---
source /workspace/kagent/k8s/install-weave-cc-plugin.sh

# --- Start Hivemind (streams CC session logs to hivemind.wandb.tools) ---
mkdir -p ~/.claude/projects
uvx --from wandb-hivemind hivemind run &
echo "=== Hivemind started (PID=$!) ==="

# --- Launch ---
cd "$KAGGLER_WORKDIR"
export IS_SANDBOX=1

if [ ! -f "$KAGGLER_PROMPT_FILE" ]; then
    echo "Missing prompt file: $KAGGLER_WORKDIR/$KAGGLER_PROMPT_FILE"
    exit 1
fi

PROMPT="You are $KAGGLER_NAME. Your branch is $BRANCH. Read $KAGGLER_PROMPT_FILE in the current directory and follow it. Go."
LOGDIR="/workspace/kagent/logs"

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    mkdir -p "$LOGDIR"
    LOGFILE="$LOGDIR/iter_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Iteration $ITERATION ($(date)) → $LOGFILE ==="

    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" --model "$AGENT_MODEL" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || true
    else
        claude -c -p "$PROMPT" --model "$AGENT_MODEL" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || \
        claude -p "$PROMPT" --model "$AGENT_MODEL" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || true
    fi

    echo "=== Restarting in 10s ==="
    sleep 10
done
