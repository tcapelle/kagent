set -e
set -o pipefail

: "${COMPETITION_DIR:?missing COMPETITION_DIR}"

BRANCH="kaggler/$KAGGLER_NAME"
AGENT_RUNTIME="${AGENT_RUNTIME:-claude}"
AGENT_MODEL="${AGENT_MODEL:-}"
KAGGLER_WORKDIR="${KAGGLER_WORKDIR:-/workspace/kagent/$COMPETITION_DIR/kaggler}"
KAGGLER_PROMPT_FILE="${KAGGLER_PROMPT_FILE:-KAGGLER_AGENT.md}"

echo "=== kagent Kaggler: $KAGGLER_NAME ==="
echo "Competition: $COMPETITION_DIR"
echo "Runtime: $AGENT_RUNTIME"
echo "Model: ${AGENT_MODEL:-default}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd /workspace/kagent

# --- Branch setup ---
git fetch origin
if git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git checkout -b "$BRANCH"
    git push -u origin "$BRANCH"
fi

# --- Install ---
uv pip install --system -e .
git config user.name "kagent-$KAGGLER_NAME"
git config user.email "kagent-$KAGGLER_NAME@kagent"

require_gh() {
    if ! command -v gh >/dev/null 2>&1; then
        echo "GitHub CLI is not present in the image"
        exit 1
    fi
}

update_claude() {
    export PATH="$HOME/.claude/bin:$HOME/.local/bin:$PATH"
    if ! command -v claude >/dev/null 2>&1; then
        echo "Claude CLI is not present in the image"
        exit 1
    fi

    claude update >/dev/null 2>&1 || echo "Claude update failed, using baked version"
}

update_codex() {
    if ! command -v codex >/dev/null 2>&1; then
        echo "Codex CLI is not present in the image"
        exit 1
    fi

    if command -v npm >/dev/null 2>&1 && npm ls -g @openai/codex >/dev/null 2>&1; then
        npm update -g @openai/codex >/dev/null 2>&1 || echo "Codex update failed, using baked version"
    else
        echo "Codex CLI is not npm-managed in this image, skipping update"
    fi
}

require_runtime_auth() {
    case "$AGENT_RUNTIME" in
        claude)
            if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
                echo "Missing ANTHROPIC_API_KEY for claude runtime"
                exit 1
            fi
            ;;
        codex)
            if [ -z "${OPENAI_API_KEY:-}" ] && [ ! -f "$HOME/.codex/auth.json" ]; then
                echo "Missing Codex authentication. Set OPENAI_API_KEY or provision ~/.codex/auth.json"
                exit 1
            fi
            # Login with API key for headless environments
            if [ -n "${OPENAI_API_KEY:-}" ]; then
                printenv OPENAI_API_KEY | codex login --with-api-key 2>/dev/null || true
            fi
            ;;
        *)
            echo "Unsupported AGENT_RUNTIME: $AGENT_RUNTIME"
            exit 1
            ;;
    esac
}

run_claude_iteration() {
    local logfile="$1"
    local args=(--model "$AGENT_MODEL" --output-format stream-json --verbose --dangerously-skip-permissions)

    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" "${args[@]}" > "$logfile" 2>&1 || true
    else
        claude -c -p "$PROMPT" "${args[@]}" > "$logfile" 2>&1 || \
        claude -p "$PROMPT" "${args[@]}" > "$logfile" 2>&1 || true
    fi
}

run_codex_iteration() {
    local logfile="$1"
    local exec_args=(exec --json --dangerously-bypass-approvals-and-sandbox --model "$AGENT_MODEL")
    local resume_args=(exec resume --last --json --dangerously-bypass-approvals-and-sandbox --model "$AGENT_MODEL")

    if [ "$ITERATION" -eq 1 ]; then
        codex "${exec_args[@]}" "$PROMPT" > "$logfile" 2>&1 || true
    else
        codex "${resume_args[@]}" "$PROMPT" > "$logfile" 2>&1 || \
        codex "${exec_args[@]}" "$PROMPT" > "$logfile" 2>&1 || true
    fi
}

case "$AGENT_RUNTIME" in
    claude)
        update_claude
        ;;
    codex)
        update_codex
        ;;
    *)
        echo "Unsupported AGENT_RUNTIME: $AGENT_RUNTIME"
        exit 1
        ;;
esac

require_gh
require_runtime_auth

# --- Launch ---
cd "$KAGGLER_WORKDIR"
export IS_SANDBOX=1

if [ ! -f "$KAGGLER_PROMPT_FILE" ]; then
    echo "Missing prompt file: $KAGGLER_WORKDIR/$KAGGLER_PROMPT_FILE"
    exit 1
fi

PROMPT="You are $KAGGLER_NAME. Your branch is $BRANCH. Read $KAGGLER_PROMPT_FILE in the current directory and follow it. Go."
LOGDIR="/workspace/kagent/logs"
mkdir -p "$LOGDIR"

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    mkdir -p "$LOGDIR"
    LOGFILE="$LOGDIR/iter_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Iteration $ITERATION ($(date)) → $LOGFILE ==="

    case "$AGENT_RUNTIME" in
        claude)
            run_claude_iteration "$LOGFILE"
            ;;
        codex)
            run_codex_iteration "$LOGFILE"
            ;;
        *)
            echo "Unsupported AGENT_RUNTIME: $AGENT_RUNTIME"
            exit 1
            ;;
    esac

    echo "=== Restarting in 10s ==="
    sleep 10
done
