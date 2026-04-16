"""Kill kagent deployments on Kubernetes.

Deletes deployments, configmaps, and optionally agent branches + PVC predictions.

Usage:
  uv run k8s/kill.py --tag mar27
  uv run k8s/kill.py --tag mar27 --clean_branches --clean_predictions
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import simple_parsing as sp
import yaml


CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_config = yaml.safe_load(CONFIG_PATH.read_text())


@dataclass
class Args:
    """Kill kagent deployments on Kubernetes."""
    tag: str = _config["tag"]  # research tag to kill
    competition: str = _config["competition"]  # competition directory
    clean_branches: bool = False  # delete agent git branches
    clean_predictions: bool = False  # delete predictions from PVC
    dry_run: bool = False


def run(cmd: list[str], dry_run: bool = False, check: bool = False) -> subprocess.CompletedProcess:
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
    return result


def main():
    args = sp.parse(Args)
    comp_label = PurePosixPath(args.competition).name

    selector = f"research-tag={args.tag},competition={comp_label}"
    print(f"Killing: {selector}")

    # Delete K8s resources
    result = run(["kubectl", "delete", "deployments,configmaps", "-l", selector], dry_run=args.dry_run)
    if not args.dry_run:
        print(result.stdout.strip() or "  No resources found")

    # Delete agent branches
    if args.clean_branches:
        print("\nCleaning git branches...")
        result = subprocess.run(
            ["git", "ls-remote", "--heads", "origin", f"refs/heads/{args.tag}/*"],
            capture_output=True, text=True,
        )
        branches = [line.split("refs/heads/")[1] for line in result.stdout.strip().splitlines() if line]
        # Also check for leaderboard branch
        lb_result = subprocess.run(
            ["git", "ls-remote", "--heads", "origin", f"refs/heads/{args.tag}-leaderboard"],
            capture_output=True, text=True,
        )
        if lb_result.stdout.strip():
            branches.append(f"{args.tag}-leaderboard")

        if branches:
            for b in branches:
                print(f"  deleting: {b}")
            if not args.dry_run:
                run(["git", "push", "origin", "--delete"] + branches)
        else:
            print("  No branches found")

    # Clean PVC predictions
    if args.clean_predictions:
        print(f"\nCleaning PVC predictions at /mnt/new-pvc/predictions/{args.tag}/...")
        pod_cmd = f"rm -rf /mnt/new-pvc/predictions/{args.tag} && echo done"
        run([
            "kubectl", "run", "pvc-cleanup", "--rm", "-it", "--restart=Never",
            "--image=busybox", f"--overrides={{"
            f'"spec":{{"containers":[{{"name":"pvc-cleanup","image":"busybox",'
            f'"command":["sh","-c","{pod_cmd}"],'
            f'"volumeMounts":[{{"name":"dataset","mountPath":"/mnt/new-pvc"}}]}}],'
            f'"volumes":[{{"name":"dataset","persistentVolumeClaim":{{"claimName":"new-pvc"}}}}]}}}}'
        ], dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
