---
name: competition-report
description: Build or refresh a kagent competition summary report — regenerate the leaderboard-evolution plot from scored commits on the shared PVC, refresh the final-leaderboard table from the organiser, and surface aha-moment quotes from the kaggler session transcripts. Use when the user asks to "update the report", "regenerate the leaderboard plot", "summarise the run", or otherwise wants the markdown summary of a competition brought up to date.
---

# Competition Report Skill

Use this skill to keep a competition summary (e.g. `<competition>/summary_<tag>.md`)
in sync with the live state of a running or finished kagent experiment.
The skill assumes the experiment is still deployed on Kubernetes and that the
organiser pod and at least some kaggler pods are reachable via `kubectl`.

## What this skill does

1. Regenerates `leaderboard_evolution.png` next to the report (staircase
   plot of per-agent running best plus a black envelope for the overall
   leader, with every scored submission as a translucent dot).
2. Refreshes the final-leaderboard table by reading the organiser's
   current `leaderboard.md` on the shared PVC.
3. Optionally surfaces aha-moment quotes from the kaggler session
   transcripts so the report can keep its "moments that mattered"
   section current.
4. Leaves the rest of the narrative (setup, harness, acknowledgements)
   untouched unless the user asks for a rewrite.

## Inputs you will need

- `tag` — the research tag (e.g. `apr16`). Defaults to the `tag` field in
  `config.yaml` at the repo root.
- `competition` — the competition directory (e.g. `gram-competition`).
  Defaults to the `competition` field in `config.yaml`.
- `report_path` — the markdown file to update. If the user does not name
  one, default to `<competition>/summary_<tag>.md`; if that file does not
  exist, propose its path before creating it.

## Workflow

### 1. Refresh the plot

Run the bundled script. It will pull `scores.json` from the organiser,
resolve commit timestamps from each agent's own pod (because commits
discarded via `git reset --hard` are only reachable in the local reflog
of the pod that made them), and render the PNG.

```bash
uv run --with matplotlib --with numpy --with simple-parsing --with pyyaml --no-project \
    .claude/skills/competition-report/scripts/build_leaderboard_plot.py \
    --tag <tag> --competition <competition>
```

You can override `--out` and `--title` when writing to a non-default
location or for a multi-tag comparison plot. Without overrides, the
plot lands at `<competition>/leaderboard_evolution.png` — the path the
summary markdown embeds.

### 2. Refresh the leaderboard table

Read the authoritative leaderboard from the organiser:

```bash
kubectl exec deployment/kagent-<tag>-organizer -- \
    cat /mnt/new-pvc/predictions/<tag>/leaderboard.md
```

Replace the markdown table under the "Final leaderboard" section of the
report with a clean version — keep the existing column set and rounding
(usually `val/l2_error`, `mae_Ux`, `mae_Uy`, `mae_Uz` to four decimals).
Drop any `unknown/...` rows that appear when the organiser scored a
submission whose branch has since been force-rewound.

### 3. Pull fresh "moments that mattered" quotes (optional)

Aha-moment quotes live in the Claude Code session logs on the shared
PVC at `/mnt/new-pvc/kagent/<tag>/<agent>/iter_*.jsonl`. Each line is a
session event; the interesting lines have `message.content[*].type ==
"text"` and tend to be short, superlative, or announce a change of rank.
Run a small `kubectl exec` with `python3` inside an agent pod to grep
for terms like `massive`, `huge`, `finally`, `beat <agent>`, `dethrone`,
`ensemble`, `surprise`. Quote the agent verbatim in italic blockquotes
under the "Aha moments" section. Attribute each quote to its agent.

Also worth scanning are each agent's `EXPERIMENT_JOURNAL.md` on the
branch — those entries follow a structured *Hypothesis / Change /
Result / Verdict / Notes* format and often contain the cleanest,
self-edited summary of a breakthrough.

### 4. Rewrite the report

Open the report file. Update in place:

- The **Final leaderboard** table (step 2).
- The **Evolution plot** caption if the phrase-of-the-day has shifted
  (e.g. new leader, new aha moment, new session count).
- The **Aha moments** section, if step 3 surfaced something not already
  cited.
- The **Setup** resource table row for `W&B project` / PR links if new
  ones exist.

Preserve the narrative, tone, and section ordering. Leave the *What the
framework demonstrates* and *Experimental harness* sections alone unless
the user explicitly asks to revise them.

### 5. Commit (only when asked)

Do not auto-commit. If the user says "commit", stage the report markdown
and the regenerated PNG together and push in a single commit with a
message that names the tag (e.g. `Refresh apr16 summary`).

## Gotchas

- **GC'd commits.** About 5–10 % of scored commits have been discarded
  from git (`git reset --hard HEAD~1`) and can't be timestamped even
  from the originating pod. The plot script skips them silently; the
  console log reports the resolved/total ratio per agent.
- **Pod availability.** Commit-timestamp resolution has to happen
  inside each agent's own pod. If an agent's pod has been deleted the
  skill can still plot the rest; warn the user about missing series
  rather than failing.
- **Shared PVC path.** The shared volume is always mounted at
  `/mnt/new-pvc` on every pod. Predictions are at
  `/mnt/new-pvc/predictions/<tag>/`, session logs at
  `/mnt/new-pvc/kagent/<tag>/<agent>/`. Ground truth is only readable
  from the organiser pod; never try to read it from a kaggler pod.
- **Plot palette.** The PNG uses a twelve-colour palette; if the
  experiment has more than twelve agents, colours will repeat. Either
  raise `PALETTE` in the script or group low-performing agents before
  plotting.

## Resources in this skill

- `scripts/build_leaderboard_plot.py` — the plot generator. Runs on the
  host machine (not in a pod); it `kubectl exec`s into pods to collect
  the data it needs.
