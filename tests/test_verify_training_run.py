import json

from tools.verify_training_run import verify_training_run


def test_verify_training_run_accepts_complete_finite_artifacts(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "step_0000001.pt").write_bytes(b"checkpoint")
    checkpoint = output / "step_0000002.pt"
    checkpoint.write_bytes(b"checkpoint")
    (output / "config.toml").write_text("[experiment]\nname = 'test'\n", encoding="utf-8")
    metrics = [
        {
            "step": 1,
            "loss": 3.0,
            "in_batch_loss": 2.0,
            "gradient_norm": 5.0,
            "logit_scale": 100.0,
            "queue_size": 2,
            "peak_cuda_allocated_bytes": 100,
        },
        {
            "step": 2,
            "loss": 2.0,
            "in_batch_loss": 1.5,
            "gradient_norm": 4.0,
            "logit_scale": 99.0,
            "queue_size": 4,
            "peak_cuda_allocated_bytes": 120,
        },
    ]
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in metrics), encoding="utf-8"
    )
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "steps": 2,
                "all_losses_finite": True,
                "all_gradients_finite": True,
                "initial_loss": 3.0,
                "final_loss": 2.0,
                "initial_in_batch_loss": 2.0,
                "final_in_batch_loss": 1.5,
                "last_gradient_norm": 4.0,
                "queue_size": 4,
                "final_checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "project_commit": "project",
                "dinov3_commit": "dinov3",
                "files": {"train_manifest": {"sha256": "manifest"}},
            }
        ),
        encoding="utf-8",
    )

    report = verify_training_run(
        output_dir=output,
        expected_steps=2,
        expected_train_manifest_sha256="manifest",
        expected_dinov3_commit="dinov3",
        expected_final_queue_size=4,
        required_checkpoint_steps=(1, 2),
        require_in_batch_loss=True,
    )

    assert report["loss"]["mean_first_window"] == 2.5
    assert report["in_batch_loss"]["last"] == 1.5
    assert report["peak_cuda_allocated_bytes"] == 120
    assert report["final_queue_size"] == 4
    assert len(report["required_checkpoints"]) == 2
