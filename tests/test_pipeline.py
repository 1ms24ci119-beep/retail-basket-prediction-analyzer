import json

from app.cli import main
from app.experiment import run_experiment
from app.visualize import create_visualizations


def test_end_to_end_experiment_writes_models_metrics_and_figures(tmp_path):
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"
    report = run_experiment(
        count=60, seed=7, epochs=1, fractions=(1.0,), data_dir=data_dir,
        artifacts_dir=artifacts_dir, results_dir=results_dir, batch_size=16, device="cpu",
    )
    assert report["split_sizes"] == {"train": 42, "validation": 9, "test": 9}
    assert (artifacts_dir / "ngram_scale_1_00.json").exists()
    assert (artifacts_dir / "transformer_scale_1_00.pt").exists()
    metrics = json.loads((results_dir / "metrics.json").read_text())
    assert metrics["training_sizes"]["1.00"]["metrics"]["ngram"]["examples"] > 0
    figures = create_visualizations(results_dir, artifacts_dir)
    assert len(figures) >= 6
    assert all(path.exists() and path.stat().st_size > 1000 for path in figures)


def test_cli_generate_data_and_missing_model_handling(tmp_path, capsys):
    data_dir = tmp_path / "generated"
    assert main(["generate-data", "--transactions", "30", "--seed", "3", "--data-dir", str(data_dir)]) == 0
    assert (data_dir / "train.jsonl").exists()
    result = main(["predict", "--basket", "bread", "--artifacts-dir", str(tmp_path / "empty")])
    assert result == 2
    assert "trained model files not found" in capsys.readouterr().err
