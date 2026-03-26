set -e
set -o pipefail

: "${COMPETITION_DIR:?missing COMPETITION_DIR}"

WORKDIR="${ORGANIZER_WORKDIR:-/workspace/kagent/$COMPETITION_DIR/organizer}"
POLL_INTERVAL=300  # 5 minutes

echo "=== kagent Organizer ==="
echo "Competition: $COMPETITION_DIR"
echo "Repo:     $REPO_URL (branch: $REPO_BRANCH)"
echo "Polling:  /mnt/new-pvc/predictions/ every ${POLL_INTERVAL}s"

cd /workspace/kagent

# Install deps from root pyproject
uv pip install --system -e .
cd "$WORKDIR"

echo "=== Organizer ready, starting scoring loop ==="

while true; do
    echo "--- Scoring check $(date) ---"
    python score.py --score_all 2>&1 || echo "  scoring error (will retry)"
    echo "--- Sleeping ${POLL_INTERVAL}s ---"
    sleep "$POLL_INTERVAL"
done
