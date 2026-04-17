set -e
set -o pipefail

: "${COMPETITION_DIR:?missing COMPETITION_DIR}"
: "${RESEARCH_TAG:?missing RESEARCH_TAG}"

REPO_DIR="/workspace/kagent"
WORKDIR="${ORGANIZER_WORKDIR:-$REPO_DIR/$COMPETITION_DIR/organizer}"
POLL_INTERVAL=60  # 1 minute — keep the leaderboard near-live for the agents
LEADERBOARD_PVC="/mnt/new-pvc/predictions/${RESEARCH_TAG}/leaderboard.md"
LEADERBOARD_BRANCH="${RESEARCH_TAG}-leaderboard"

echo "=== kagent Organizer ==="
echo "Tag: $RESEARCH_TAG"
echo "Competition: $COMPETITION_DIR"
echo "Polling every ${POLL_INTERVAL}s"

cd "$REPO_DIR"
uv pip install --system -e .
git config user.name "kagent-organizer"
git config user.email "kagent-organizer@kagent"

push_leaderboard() {
    [ -f "$LEADERBOARD_PVC" ] || return 0
    # Sync the local ref with origin first so a restarted pod builds on top
    # of what's already published, not on its own empty history.
    git fetch origin "$LEADERBOARD_BRANCH:refs/heads/$LEADERBOARD_BRANCH" >/dev/null 2>&1 || true

    cp "$LEADERBOARD_PVC" /tmp/leaderboard.md
    TREE=$(git hash-object -w /tmp/leaderboard.md)
    NEW_TREE=$(printf "100644 blob %s\tleaderboard.md\n" "$TREE" | git mktree)

    if git rev-parse --verify "refs/heads/$LEADERBOARD_BRANCH" >/dev/null 2>&1; then
        PARENT=$(git rev-parse "refs/heads/$LEADERBOARD_BRANCH")
        OLD_TREE=$(git rev-parse "$PARENT^{tree}")
        [ "$NEW_TREE" = "$OLD_TREE" ] && return 0
        COMMIT=$(echo "Update leaderboard" | git commit-tree "$NEW_TREE" -p "$PARENT")
    else
        COMMIT=$(echo "Update leaderboard" | git commit-tree "$NEW_TREE")
    fi

    git update-ref "refs/heads/$LEADERBOARD_BRANCH" "$COMMIT"
    # Force-with-lease: the leaderboard is single-writer state, so a diverging
    # remote means another organizer pod is alive — fall back to a hard force.
    git push origin "$LEADERBOARD_BRANCH" 2>/dev/null \
        || git push --force-with-lease origin "$LEADERBOARD_BRANCH" 2>/dev/null \
        || git push --force origin "$LEADERBOARD_BRANCH" 2>/dev/null \
        || echo "  Leaderboard push failed"
}

echo "=== Organizer ready ==="

while true; do
    echo "--- Scoring check $(date) ---"
    python "$WORKDIR/score.py" --score_all 2>&1 || echo "  scoring error (will retry)"
    push_leaderboard
    echo "--- Sleeping ${POLL_INTERVAL}s ---"
    sleep "$POLL_INTERVAL"
done
