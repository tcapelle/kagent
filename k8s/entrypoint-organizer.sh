set -e
set -o pipefail

: "${COMPETITION_DIR:?missing COMPETITION_DIR}"
: "${RESEARCH_TAG:?missing RESEARCH_TAG}"

REPO_DIR="/workspace/kagent"
WORKDIR="${ORGANIZER_WORKDIR:-$REPO_DIR/$COMPETITION_DIR/organizer}"
POLL_INTERVAL=300  # 5 minutes
LEADERBOARD_BRANCH="${RESEARCH_TAG}-leaderboard"
LEADERBOARD_PVC="/mnt/new-pvc/predictions/${RESEARCH_TAG}/leaderboard.md"

echo "=== kagent Organizer ==="
echo "Tag: $RESEARCH_TAG"
echo "Competition: $COMPETITION_DIR"
echo "Leaderboard branch: $LEADERBOARD_BRANCH"
echo "Polling every ${POLL_INTERVAL}s"

cd "$REPO_DIR"
uv pip install --system -e .

# Git setup
git config user.name "kagent-organizer"
git config user.email "kagent-organizer@kagent"
git fetch origin
if git rev-parse --verify "origin/$LEADERBOARD_BRANCH" >/dev/null 2>&1; then
    git checkout -B "$LEADERBOARD_BRANCH" "origin/$LEADERBOARD_BRANCH"
else
    git checkout -B "$LEADERBOARD_BRANCH" "origin/main"
    git push -u origin "$LEADERBOARD_BRANCH"
fi

push_leaderboard() {
    [ -f "$LEADERBOARD_PVC" ] || return 0
    cp "$LEADERBOARD_PVC" "$REPO_DIR/leaderboard.md"
    git -C "$REPO_DIR" add leaderboard.md
    git -C "$REPO_DIR" diff --cached --quiet && return 0
    git -C "$REPO_DIR" commit -m "Update leaderboard"
    git -C "$REPO_DIR" push origin "$LEADERBOARD_BRANCH" || echo "  Leaderboard push failed"
}

echo "=== Organizer ready ==="

while true; do
    echo "--- Scoring check $(date) ---"
    python "$WORKDIR/score.py" --score_all 2>&1 || echo "  scoring error (will retry)"
    push_leaderboard
    echo "--- Sleeping ${POLL_INTERVAL}s ---"
    sleep "$POLL_INTERVAL"
done
