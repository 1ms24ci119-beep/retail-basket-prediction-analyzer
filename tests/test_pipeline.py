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


def test_standalone_cli_training_evaluation_comparison_and_prediction(tmp_path, capsys):
    data_dir = tmp_path / "sessions"
    artifacts_dir = tmp_path / "models"
    results_dir = tmp_path / "reports"
    assert main(["generate-data", "--transactions", "60", "--data-dir", str(data_dir)]) == 0
    assert main([
        "train-ngram", "--data-dir", str(data_dir), "--artifacts-dir", str(artifacts_dir),
    ]) == 0
    assert main([
        "train-transformer", "--data-dir", str(data_dir), "--artifacts-dir", str(artifacts_dir),
        "--epochs", "1", "--batch-size", "16", "--device", "cpu",
    ]) == 0
    assert main([
        "evaluate", "--data-dir", str(data_dir), "--artifacts-dir", str(artifacts_dir),
        "--results-dir", str(results_dir),
    ]) == 0
    assert main(["compare", "--results-dir", str(results_dir)]) == 0
    assert main([
        "predict", "--basket", "bread,butter", "--artifacts-dir", str(artifacts_dir), "--top-k", "2",
    ]) == 0
    output = capsys.readouterr().out
    assert "cross_entropy" in output
    assert "snack_jam" in output
    assert (results_dir / "model_comparison.csv").exists()
