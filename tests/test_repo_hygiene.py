import subprocess
from pathlib import Path


def test_env_and_model_files_are_ignored_but_example_config_is_trackable():
    assert subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", ".env"], check=False
    ).returncode == 0
    assert subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "artifacts/model.pt"], check=False
    ).returncode == 0
    assert subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", ".env.example"], check=False
    ).returncode == 1


def test_example_environment_contains_no_api_keys():
    content = Path(".env.example").read_text(encoding="utf-8")
    assert "API_KEY=" not in content
    assert "your-key-here" not in content
    assert "DATA_DIR=data/generated" in content
