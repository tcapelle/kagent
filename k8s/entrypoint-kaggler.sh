set -e
set -o pipefail

: "${COMPETITION_REPO:?missing COMPETITION_REPO}"
: "${KAGGLER_NAME:?missing KAGGLER_NAME}"
: "${RESEARCH_TAG:?missing RESEARCH_TAG}"
: "${AGENT_MODEL:?missing AGENT_MODEL}"

BRANCH="$RESEARCH_TAG/kaggler/$KAGGLER_NAME"
KAGGLER_WORKDIR=/workspace/competition/kaggler
KAGGLER_PROMPT_FILE=KAGGLER_AGENT.md

echo "=== kagent Kaggler: $KAGGLER_NAME ==="
echo "Competition: $COMPETITION_REPO"
echo "Model: $AGENT_MODEL"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd /workspace/competition

# --- Branch setup in the competition repo (sparse: kaggler dir only, no organizer) ---
if git ls-remote --exit-code origin "refs/heads/$BRANCH" >/dev/null 2>&1; then
    git fetch origin "$BRANCH:refs/remotes/origin/$BRANCH"
    git checkout -b "$BRANCH" "origin/$BRANCH"
else
    git checkout -b "$BRANCH"
    git push -u origin "$BRANCH"
fi
git sparse-checkout set kaggler .agents

# --- Install (shared deps live in kagent) ---
uv pip install --system -e /workspace/kagent
git config user.name "kagent-$KAGGLER_NAME"
git config user.email "kagent-$KAGGLER_NAME@kagent"

# --- pi packages + settings (Exa search, subagents, Weave tracing) ---
npm install --prefix /workspace/kagent/k8s/pi-weave --no-package-lock --no-audit --no-fund
mkdir -p ~/.pi/agent
cp /workspace/kagent/k8s/pi-settings.json ~/.pi/agent/settings.json
pi list  # installs the packages declared in settings.json
pi auth check --model "$AGENT_MODEL"

# --- Start Hivemind (streams pi session logs to hivemind.wandb.tools) ---
mkdir -p ~/.pi/agent/sessions
uvx --from wandb-hivemind hivemind run &
echo "=== Hivemind started (PID=$!) ==="

# --- Launch ---
cd "$KAGGLER_WORKDIR"

if [ ! -f "$KAGGLER_PROMPT_FILE" ]; then
    echo "Missing prompt file: $KAGGLER_WORKDIR/$KAGGLER_PROMPT_FILE"
    exit 1
fi

PROMPT="You are $KAGGLER_NAME. Your branch is $BRANCH. Read $KAGGLER_PROMPT_FILE in the current directory and follow it. You may delegate to subagents whenever it helps (e.g. researcher for literature search, reviewer before committing). Go."
LOGDIR="/mnt/new-pvc/kagent/$RESEARCH_TAG/$KAGGLER_NAME"

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    mkdir -p "$LOGDIR"
    LOGFILE="$LOGDIR/iter_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Iteration $ITERATION ($(date)) → $LOGFILE ==="

    # stdout is pure JSONL events; pi diagnostics go to stderr (pod log).
    # pi -p also reads piped stdin, so close it or the run waits forever.
    pi -p --mode json --approve --model "$AGENT_MODEL" "$PROMPT" < /dev/null > "$LOGFILE" || true

    echo "=== Restarting in 10s ==="
    sleep 10
done
