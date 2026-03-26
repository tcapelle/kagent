"""Launch kagent kaggler pods on Kubernetes."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import simple_parsing as sp

TEMPLATES_DIR = Path(__file__).parent
KAGGLER_TEMPLATE = TEMPLATES_DIR / "kaggler-deployment.yaml"
ORGANIZER_TEMPLATE = TEMPLATES_DIR / "organizer-deployment.yaml"
PREPARE_TEMPLATE = TEMPLATES_DIR / "prepare-splits-job.yaml"
SUPPORTED_AGENT_RUNTIMES = {"claude", "codex"}
DEFAULT_AGENT_MODELS = {
    "claude": "claude-opus-4-6[1m]",
    "codex": "gpt-5.4",
}

KAGGLER_NAMES = [
    "frieren", "fern", "tanjiro", "nezuko", "alphonse", "edward",
    "thorfinn", "askeladd", "violet", "gilbert", "senku", "kohaku",
    "emma", "norman", "chihiro", "haku", "shoya", "shouko",
    "mitsuha", "taki", "shinji", "rei", "kaneda", "tetsuo",
]


@dataclass
class Args:
    """Launch kagent kaggler pods on Kubernetes."""
    tag: str  # research tag (e.g. mar18)
    competition: str = "cfd-competition"  # repo-relative competition directory
    agent_runtime: str = "claude"  # one of: claude, codex
    agent_model: str = ""  # defaults depend on agent_runtime
    names: str = ""  # comma-separated kaggler names (e.g. "frieren,fern")
    n_kagglers: int = 4  # number of kagglers (ignored if --names provided)
    repo_url: str = "https://github.com/tcapelle/kagent.git"
    repo_branch: str = "main"
    image: str = "ghcr.io/tcapelle/dev_box:477a81a"
    wandb_entity: str = "wandb-applied-ai-team"
    wandb_project: str = "kagent-v1"
    organizer: bool = False  # deploy the organizer (scoring loop)
    prepare: bool = False  # run prepare_splits.py one-shot job
    dry_run: bool = False


def render_template(template: str, replacements: dict[str, str]) -> str:
    out = template
    for key, value in replacements.items():
        out = out.replace(f"{{{{{key}}}}}", value)
    return out


def render_configmap(name: str, labels: dict[str, str], data: dict[str, str]) -> str:
    lines = ["apiVersion: v1", "kind: ConfigMap", "metadata:", f"  name: {name}", "  labels:"]
    for k, v in labels.items():
        lines.append(f"    {k}: {v}")
    lines.append("data:")
    for k, v in data.items():
        lines.append(f"  {k}: \"{v}\"")
    return "\n".join(lines)


def kubectl_apply(manifest: str, name: str):
    print(f"Launching: {name}")
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest, text=True, capture_output=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"  {result.stdout.strip()}")


def normalize_competition_dir(competition: str) -> str:
    """Normalize and validate the repo-relative competition directory."""
    competition_dir = PurePosixPath(competition.strip())
    if not competition_dir.parts:
        raise ValueError("competition must not be empty")
    if competition_dir.is_absolute():
        raise ValueError("competition must be repo-relative, not absolute")
    if ".." in competition_dir.parts:
        raise ValueError("competition must not contain '..'")
    return str(competition_dir)


def build_competition_env(competition_dir: str) -> dict[str, str]:
    """Build shared environment variables for competition-specific runtimes."""
    repo_root = PurePosixPath("/workspace/kagent")
    comp_path = PurePosixPath(competition_dir)
    comp_root = repo_root / comp_path
    return {
        "COMPETITION_DIR": str(comp_path),
        "COMPETITION_NAME": comp_path.name,
        "KAGGLER_WORKDIR": str(comp_root / "kaggler"),
        "ORGANIZER_WORKDIR": str(comp_root / "organizer"),
        "KAGGLER_PROMPT_FILE": "KAGGLER_AGENT.md",
    }


def normalize_agent_runtime(agent_runtime: str) -> str:
    runtime = agent_runtime.strip().lower()
    if runtime not in SUPPORTED_AGENT_RUNTIMES:
        valid = ", ".join(sorted(SUPPORTED_AGENT_RUNTIMES))
        raise ValueError(f"agent_runtime must be one of: {valid}")
    return runtime


def resolve_agent_model(agent_runtime: str, agent_model: str) -> str:
    model = agent_model.strip()
    return model or DEFAULT_AGENT_MODELS[agent_runtime]


def main():
    args = sp.parse(Args)
    competition_dir = normalize_competition_dir(args.competition)
    agent_runtime = normalize_agent_runtime(args.agent_runtime)
    agent_model = resolve_agent_model(agent_runtime, args.agent_model)
    competition_env = build_competition_env(competition_dir)

    # --- Prepare splits job ---
    if args.prepare:
        configmap = render_configmap(
            name="kagent-config-prepare",
            labels={"app": "kagent", "role": "prepare", "research-tag": args.tag},
            data={
                "REPO_URL": args.repo_url,
                "REPO_BRANCH": args.repo_branch,
                **competition_env,
            },
        )
        job = render_template(PREPARE_TEMPLATE.read_text(), {
            "RESEARCH_TAG": args.tag, "IMAGE": args.image,
        })
        manifest = configmap + "\n---\n" + job
        if args.dry_run:
            print(manifest)
        else:
            kubectl_apply(manifest, "prepare-splits")
            print(f"\n  kubectl logs -f job/kagent-prepare-splits")
        return

    # --- Resolve kaggler list ---
    if args.names:
        kaggler_list = [n.strip() for n in args.names.split(",")]
    else:
        kaggler_list = KAGGLER_NAMES[:args.n_kagglers]

    # --- Deploy kagglers ---
    template = KAGGLER_TEMPLATE.read_text()
    for name in kaggler_list:
        configmap = render_configmap(
            name=f"kagent-config-kaggler-{name}",
            labels={"app": "kagent", "role": "kaggler", "research-tag": args.tag},
            data={
                "REPO_URL": args.repo_url,
                "REPO_BRANCH": args.repo_branch,
                "KAGGLER_NAME": name,
                "RESEARCH_TAG": args.tag,
                "WANDB_ENTITY": args.wandb_entity,
                "WANDB_PROJECT": args.wandb_project,
                "WANDB_MODE": "online",
                "AGENT_RUNTIME": agent_runtime,
                "AGENT_MODEL": agent_model,
                **competition_env,
            },
        )
        deployment = render_template(template, {
            "KAGGLER_NAME": name,
            "RESEARCH_TAG": args.tag,
            "IMAGE": args.image,
        })
        manifest = configmap + "\n---\n" + deployment
        if args.dry_run:
            print(f"--- {name} ---")
            print(manifest)
            print()
        else:
            kubectl_apply(manifest, f"kaggler {name}")

    # --- Deploy organizer ---
    if args.organizer:
        configmap = render_configmap(
            name="kagent-config-organizer",
            labels={"app": "kagent", "role": "organizer", "research-tag": args.tag},
            data={
                "REPO_URL": args.repo_url,
                "REPO_BRANCH": args.repo_branch,
                "RESEARCH_TAG": args.tag,
                "WANDB_ENTITY": args.wandb_entity,
                "WANDB_PROJECT": args.wandb_project,
                **competition_env,
            },
        )
        deployment = render_template(ORGANIZER_TEMPLATE.read_text(), {
            "RESEARCH_TAG": args.tag,
            "IMAGE": args.image,
        })
        manifest = configmap + "\n---\n" + deployment
        if args.dry_run:
            print("--- Organizer ---")
            print(manifest)
            print()
        else:
            kubectl_apply(manifest, "organizer")

    if not args.dry_run:
        print(f"\nLaunched {len(kaggler_list)} kagglers: {', '.join(kaggler_list)}")
        print(f"Competition: {competition_dir}")
        print(f"Agent runtime: {agent_runtime} ({agent_model})")
        print(f"Each on branch: kaggler/<name>")
        print(f"Predictions: /mnt/new-pvc/predictions/<name>/<commit>/")
        if args.organizer:
            print(f"Organizer: scoring every 5 min")
        print(f"\nMonitor:")
        print(f"  kubectl get deployments -l research-tag={args.tag}")
        print(f"  kubectl logs -f deployment/kagent-{kaggler_list[0]}")
        if args.organizer:
            print(f"  kubectl logs -f deployment/kagent-organizer")
        print(f"\nStop:")
        print(f"  kubectl delete deployments,configmaps -l research-tag={args.tag}")


if __name__ == "__main__":
    main()
