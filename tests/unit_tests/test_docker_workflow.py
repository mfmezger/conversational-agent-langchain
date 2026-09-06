"""Static safety checks for the Docker image workflow."""

from pathlib import Path

import yaml

WORKFLOW_PATH = Path(__file__).parents[2] / ".github" / "workflows" / "docker.yml"
DOCKERIGNORE_PATH = Path(__file__).parents[2] / ".dockerignore"
FORK_CONDITION = "github.event_name == 'pull_request' && github.event.pull_request.head.repo.full_name != github.repository"


def load_build_job() -> dict:
    """Load the canonical Docker validation job."""
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert list(workflow["jobs"]) == ["build"]
    return workflow["jobs"]["build"]


def test_fork_pr_fails_before_checkout() -> None:
    """Fork PRs must fail in a no-secret step before checkout."""
    steps = load_build_job()["steps"]
    gate = steps[0]

    assert gate["name"] == "Reject fork pull requests before validation"
    assert gate["if"] == FORK_CONDITION
    assert "exit 1" in gate["run"]
    assert "uses" not in gate
    assert "env" not in gate
    assert "secrets." not in gate["run"]
    assert steps[1]["uses"].startswith("actions/checkout@")
    assert all("if" not in step for step in steps[1:])


def test_trusted_pr_reaches_build_steps() -> None:
    """The canonical job and trusted build path must not be skipped."""
    job = load_build_job()
    steps = job["steps"]

    assert "if" not in job
    for name in ("Build backend image", "Build frontend image"):
        step = next(step for step in steps if step.get("name") == name)
        assert "if" not in step


def test_dependabot_reaches_credential_preflight() -> None:
    """Same-repository bots must use the standard secret preflight."""
    steps = load_build_job()["steps"]
    gate = steps[0]
    preflight = next(step for step in steps if step.get("name") == "Verify DHI credentials are configured")

    assert "github.actor" not in gate["if"]
    assert "if" not in preflight
    assert preflight["env"] == {
        "DOCKER_USERNAME": "${{ secrets.DOCKER_USERNAME }}",
        "DOCKER_PASSWORD": "${{ secrets.DOCKER_PASSWORD }}",
    }
    assert "Dependabot secrets" in preflight["run"]


def test_checkout_does_not_persist_credentials() -> None:
    """Do not leave checkout credentials in the Docker build workspace."""
    steps = load_build_job()["steps"]
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))

    assert checkout.get("with", {}).get("persist-credentials") is False


def test_docker_context_excludes_git_metadata() -> None:
    """Exclude Git metadata whether .git is a directory or a worktree file."""
    patterns = DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()

    assert ".git" in patterns
