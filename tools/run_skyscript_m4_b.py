#!/usr/bin/env python3
"""Run one restart-safe staged Web/SAT experiment in SkyScript M4-B."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dinotxt_rs.training.provenance import git_commit, sha256_file  # noqa: E402
from tools.verify_preflight_compatibility import compare_preflights  # noqa: E402
from tools.verify_skyscript_m4_b_configs import (  # noqa: E402
    BACKBONES,
    config_path,
    experiment_name,
    load_config_matrix,
    verify_config_matrix,
)

STAGES = (100, 250, 500)
TRAIN_SHA = "4264d884d564631816baf3e339e7ca12be2958966d99a60f6a75ae3d5d4dbaec"
VAL_SHA = "062238716b8fddad890b4f97354588ad7524f1099667d3c0baa932002845b7f6"
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
RSICD_ANNOTATION_SHA = "5e342037d469d074711676bdb9c02b6942a624530b1959d24d2734e68af9cede"
M4_A_SUMMARY_SHA = "69bef90195afca5d19bd79a110f2261c0ece9a281fe6b71672aaa3d93ffaa9dc"
WEIGHT_SHA = {
    "web": "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035",
    "sat": "eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48",
}
DINOTXT_SHA = "a442d8f52a3a7ad715bf6b7d8117fb3a84d54249389b0a13f6956cd0d2eca4f0"
BPE_SHA = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("web", "sat"))
    parser.add_argument("--seed", required=True, type=int, choices=(23, 47))
    parser.add_argument("--stop-after-stage", type=int, choices=STAGES, default=100)
    parser.add_argument("--rsicd-root", type=Path, default=Path("assets/data/raw/rsicd"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--skip-code-checks",
        action="store_true",
        help="Only for the matrix wrapper, which runs the checks once before all jobs.",
    )
    return parser.parse_args()


def _run(args: list[str], *, log: Path | None = None) -> None:
    if log is None:
        subprocess.run(args, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        log = log.with_name(f"{log.stem}.attempt-{timestamp}{log.suffix}")
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, args)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required M4-B input: {path}")


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"Unexpected {label}: expected {expected}, got {observed}")


def _copy_immutable(source: Path, destination: Path) -> None:
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise FileExistsError(f"Existing artifact differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class M4BRun:
    def __init__(self, args: argparse.Namespace, root: Path) -> None:
        self.args = args
        self.root = root
        self.domain = args.domain
        self.seed = args.seed
        self.name = experiment_name(self.domain, self.seed)
        self.config = config_path(Path("configs"), self.domain, self.seed)
        self.run_dir = Path("outputs") / self.name
        self.gate_dir = Path("outputs") / f"skyscript_gate_m4_b_{self.domain}_seed{self.seed}"
        self.train_manifest = Path(
            "assets/data/manifests/"
            "skyscript_images23_train_raw_unique36495_seed11_global77.jsonl"
        )
        self.val_manifest = Path(
            "assets/data/manifests/"
            "skyscript_images23_val_raw_unique4055_seed23_global77.jsonl"
        )
        self.m4_a_summary = Path("outputs/skyscript_m4_rq1_seed11/summary.json")
        self.weights = Path(BACKBONES[self.domain])
        self.dinotxt = Path(
            "assets/checkpoints/"
            "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
        )
        self.bpe = Path("assets/checkpoints/bpe_simple_vocab_16e6.txt.gz")
        self.rsicd_annotation = args.rsicd_root / "dataset_rsicd.json"
        self.rsicd_manifest = Path("assets/data/manifests/rsicd_val_retrieval_v1.jsonl")
        self.rsicd_audit = Path("assets/data/manifests/rsicd_val_retrieval_v1.audit.json")

    def preflight(self) -> None:
        status = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout
        if status:
            raise RuntimeError(
                "Refusing M4-B with uncommitted or untracked project changes:\n" + status
            )
        verify_config_matrix(load_config_matrix(Path("configs")))
        required = (
            self.config,
            self.m4_a_summary,
            self.train_manifest,
            self.val_manifest,
            self.weights,
            self.dinotxt,
            self.bpe,
            self.rsicd_annotation,
        )
        for path in required:
            _require_file(path)
        _require_hash(self.m4_a_summary, M4_A_SUMMARY_SHA, "M4-A summary SHA-256")
        _require_hash(self.train_manifest, TRAIN_SHA, "training manifest SHA-256")
        _require_hash(self.val_manifest, VAL_SHA, "validation manifest SHA-256")
        _require_hash(self.weights, WEIGHT_SHA[self.domain], "backbone weights SHA-256")
        _require_hash(self.dinotxt, DINOTXT_SHA, "dino.txt weights SHA-256")
        _require_hash(self.bpe, BPE_SHA, "BPE vocabulary SHA-256")
        _require_hash(
            self.rsicd_annotation, RSICD_ANNOTATION_SHA, "RSICD annotation SHA-256"
        )
        if git_commit(Path("external/dinov3")) != DINOV3_COMMIT:
            raise ValueError("Unexpected DINOv3 upstream commit")
        report = json.loads(self.m4_a_summary.read_text(encoding="utf-8"))
        if report.get("gate") != "SkyScript_M4_RQ1_seed11" or report.get("status") != "pass":
            raise ValueError("M4-A prerequisite is not the frozen passing report")
        print("m4_a_prerequisite=verified m4_b_config_matrix=verified")
        if not self.args.skip_code_checks:
            print("Running code checks once before GPU work...")
            _run(["ruff", "check", "."])
            _run([sys.executable, "-m", "pytest"])
            _run([sys.executable, "-m", "compileall", "-q", "src", "tools"])

    def prepare_shared_evidence(self) -> None:
        self.gate_dir.mkdir(parents=True, exist_ok=True)
        (self.gate_dir / "quarantine").mkdir(exist_ok=True)
        if not self.rsicd_manifest.exists() and not self.rsicd_audit.exists():
            print("Preparing RSICD-val retrieval manifest...")
            _run(
                [
                    sys.executable,
                    "tools/prepare_rsicd_retrieval_manifest.py",
                    "--annotations",
                    str(self.rsicd_annotation),
                    "--images-root",
                    str(self.args.rsicd_root),
                    "--split",
                    "val",
                    "--output",
                    str(self.rsicd_manifest),
                    "--audit-output",
                    str(self.rsicd_audit),
                ]
            )
        elif not self.rsicd_manifest.is_file() or not self.rsicd_audit.is_file():
            raise FileNotFoundError("RSICD-val manifest and audit must exist together")
        audit = json.loads(self.rsicd_audit.read_text(encoding="utf-8"))
        rsicd_sha = sha256_file(self.rsicd_manifest)
        if (
            audit.get("dataset") != "RSICD"
            or audit.get("split") != "val"
            or audit.get("manifest", {}).get("sha256") != rsicd_sha
            or audit.get("images", 0) <= 0
            or audit.get("captions", 0) <= 0
        ):
            raise ValueError("RSICD-val manifest audit is invalid")
        self._write_preflight(rsicd_sha)
        self._write_runner_history()
        train_val = self.gate_dir / "train_vs_val_overlap.json"
        shared = Path("outputs/skyscript_s0_train_val_overlap.json")
        if not train_val.exists():
            if not shared.exists():
                _run(
                    [
                        sys.executable,
                        "tools/audit_manifest_image_overlap.py",
                        "--left-manifest",
                        str(self.train_manifest),
                        "--right-manifest",
                        str(self.val_manifest),
                        "--output",
                        str(shared),
                        "--require-zero-overlap",
                    ],
                    log=self.gate_dir / "train_vs_val_overlap.log",
                )
            _copy_immutable(shared, train_val)
        train_rsicd = self.gate_dir / "train_vs_rsicd_val_overlap.json"
        if not train_rsicd.exists():
            _run(
                [
                    sys.executable,
                    "tools/audit_manifest_image_overlap.py",
                    "--left-manifest",
                    str(self.train_manifest),
                    "--right-manifest",
                    str(self.rsicd_manifest),
                    "--output",
                    str(train_rsicd),
                    "--require-zero-overlap",
                ],
                log=self.gate_dir / "train_vs_rsicd_val_overlap.log",
            )

    def _preflight_values(self, rsicd_sha: str) -> dict[str, str]:
        return {
            "project_commit": git_commit(self.root) or "unknown",
            "m4_a_summary": str(self.m4_a_summary),
            "m4_a_summary_sha256": M4_A_SUMMARY_SHA,
            "config": str(self.config),
            "config_sha256": sha256_file(self.config),
            "training_output": str(self.run_dir),
            "backbone_domain": self.domain,
            "backbone_weights_sha256": WEIGHT_SHA[self.domain],
            "dinotxt_weights_sha256": DINOTXT_SHA,
            "bpe_vocab_sha256": BPE_SHA,
            "train_manifest_sha256": TRAIN_SHA,
            "skyscript_val_manifest_sha256": VAL_SHA,
            "rsicd_val_manifest_sha256": rsicd_sha,
            "seed": str(self.seed),
            "stages": "100,250,500",
            "validation_every": "50",
            "warmup_steps": "50",
            "batch_size": str(self.args.batch_size),
            "num_workers": str(self.args.num_workers),
            "retrieval_chunk_size": str(self.args.chunk_size),
            "retention_margin_absolute_mean_recall": "0.01",
        }

    def _write_preflight(self, rsicd_sha: str) -> None:
        path = self.gate_dir / "preflight.txt"
        current = self._preflight_values(rsicd_sha)
        encoded = "".join(f"{key}={value}\n" for key, value in current.items())
        if path.exists():
            stored = dict(
                line.split("=", 1)
                for line in path.read_text(encoding="utf-8").splitlines()
            )
            result = compare_preflights(stored, current)
            if result["advisory_fields"]:
                print(
                    "WARNING: project commit changed; frozen protocol fields still match. "
                    "Continuing and recording this invocation.",
                    file=sys.stderr,
                )
            if result["blocking_fields"]:
                raise ValueError(
                    "Existing M4-B artifacts differ in frozen preflight fields: "
                    f"{result['blocking_fields']}"
                )
            return
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        os.replace(temporary, path)

    def _write_runner_history(self) -> None:
        preflight = dict(
            line.split("=", 1)
            for line in (self.gate_dir / "preflight.txt").read_text().splitlines()
        )
        initial = preflight.get("project_commit")
        current = git_commit(self.root)
        changed: list[str] = []
        if initial and current and initial != current:
            diff = subprocess.run(
                ["git", "diff", "--name-only", f"{initial}..{current}"],
                check=False,
                capture_output=True,
                text=True,
            )
            if diff.returncode == 0:
                changed = diff.stdout.splitlines()
        record = {
            "format_version": 1,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
            "requested_stage": self.args.stop_after_stage,
            "initial_project_commit": initial,
            "current_project_commit": current,
            "project_commit_match": initial == current,
            "changed_files_since_initial_commit": changed,
        }
        with (self.gate_dir / "runner_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def current_step(self) -> int:
        summary_path = self.run_dir / "training_summary.json"
        if not summary_path.is_file():
            partial_names = ("config.toml", "provenance.json", "metrics.jsonl")
            if any((self.run_dir / name).exists() for name in partial_names) or list(
                self.run_dir.glob("step_*.pt")
            ):
                raise RuntimeError(
                    f"Partial M4-B artifacts lack a training summary: {self.run_dir}"
                )
            return 0
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        state = (summary.get("steps"), summary.get("completed"))
        if state not in {(100, False), (250, False), (500, True)}:
            raise ValueError(f"Unexpected M4-B training state: {state}")
        if summary.get("target_steps") != 500:
            raise ValueError("M4-B training target is not 500")
        return int(summary["steps"])

    def _validate_resume_tail(self, step: int) -> None:
        if not (self.run_dir / f"step_{step:07d}.pt").is_file():
            raise FileNotFoundError(f"Missing resume checkpoint at step {step}")
        if list(self.run_dir.glob("*.part")):
            raise ValueError("Incomplete atomic output prevents safe M4-B resume")
        if (self.run_dir / "config.toml").read_bytes() != self.config.read_bytes():
            raise ValueError("Run config snapshot differs from the immutable M4-B config")
        metrics = [
            json.loads(line)
            for line in (self.run_dir / "metrics.jsonl").read_text().splitlines()
        ]
        if [record.get("step") for record in metrics] != list(range(1, step + 1)):
            raise ValueError("metrics.jsonl does not end at the resume checkpoint")
        validation = [
            json.loads(line)
            for line in (self.run_dir / "validation.jsonl").read_text().splitlines()
        ]
        if [record.get("step") for record in validation] != list(range(0, step + 1, 50)):
            raise ValueError("validation.jsonl does not end at the resume checkpoint")
        history_path = self.run_dir / "resume_history.jsonl"
        history = (
            [json.loads(line) for line in history_path.read_text().splitlines()]
            if history_path.is_file()
            else []
        )
        expected = [] if step == 100 else [100]
        if [record.get("checkpoint_step") for record in history] != expected:
            raise ValueError("M4-B resume history does not match the staged protocol")

    def train_to_stage(self, target: int) -> None:
        observed = self.current_step()
        if observed >= target:
            print(f"Reusing {self.domain} seed{self.seed} training through step {observed}.")
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if observed == 0 and target == 100:
            _run(
                [
                    sys.executable,
                    "-m",
                    "dinotxt_rs.cli.smoke_model",
                    "--config",
                    str(self.config),
                    "--batch-size",
                    "16",
                ],
                log=self.run_dir / "smoke.log",
            )
            command = [
                sys.executable,
                "-m",
                "dinotxt_rs.cli.train",
                "--config",
                str(self.config),
                "--stop-after-step",
                "100",
            ]
            log = self.run_dir / "train_phase_000_to_100.log"
        elif observed == 100 and target == 250:
            self._validate_resume_tail(100)
            command = [
                sys.executable,
                "-m",
                "dinotxt_rs.cli.train",
                "--config",
                str(self.config),
                "--resume",
                str(self.run_dir / "step_0000100.pt"),
                "--stop-after-step",
                "250",
            ]
            log = self.run_dir / "train_phase_100_to_250.log"
        elif observed == 250 and target == 500:
            self._validate_resume_tail(250)
            command = [
                sys.executable,
                "-m",
                "dinotxt_rs.cli.train",
                "--config",
                str(self.config),
                "--resume",
                str(self.run_dir / "step_0000250.pt"),
            ]
            log = self.run_dir / "train_phase_250_to_500.log"
        else:
            raise RuntimeError(f"Cannot progress M4-B from step {observed} to {target}")
        _run(command, log=log)
        if self.current_step() != target:
            raise RuntimeError(f"Training did not stop at required step {target}")

    def _ensure_training_report(self, stage: int) -> Path:
        report = self.gate_dir / f"training_step{stage}.json"
        if report.exists():
            if stage == 500:
                _copy_immutable(report, self.run_dir / "verification_report.json")
            return report
        if self.current_step() != stage:
            raise RuntimeError(f"Cannot verify step{stage} after training advanced")
        args = [
            sys.executable,
            "tools/verify_training_run.py",
            "--output",
            str(self.run_dir),
            "--expected-steps",
            str(stage),
            "--expected-target-steps",
            "500",
            "--expected-train-manifest-sha256",
            TRAIN_SHA,
            "--expected-val-manifest-sha256",
            VAL_SHA,
            "--expected-dinov3-commit",
            DINOV3_COMMIT,
            "--expected-final-queue-size",
            "0",
            "--require-validation",
            "--validation-every",
            "50",
            "--require-best-checkpoint",
            "--expected-validation-loss-batch-size",
            "16",
            "--expected-validation-forward-batch-size",
            "64",
        ]
        for checkpoint in range(0, stage + 1, 50):
            args.extend(("--required-checkpoint-step", str(checkpoint)))
        if stage == 100:
            args.extend(("--require-incomplete", "--forbid-resume"))
        elif stage == 250:
            args.extend(("--require-incomplete", "--required-resume-step", "100"))
        else:
            args.extend(
                (
                    "--require-completed",
                    "--required-resume-step",
                    "100",
                    "--required-resume-step",
                    "250",
                )
            )
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.gate_dir, delete=False
        ) as handle:
            temporary = Path(handle.name)
            process = subprocess.run(args, stdout=handle, text=True)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, args)
        os.replace(temporary, report)
        if stage == 500:
            _copy_immutable(report, self.run_dir / "verification_report.json")
        return report

    def _ensure_evidence(self, stage: int) -> None:
        parity = self.run_dir / "skyscript_step0_parity.json"
        if not parity.exists():
            _run(
                [
                    sys.executable,
                    "-m",
                    "dinotxt_rs.cli.check_step0_parity",
                    "--config",
                    str(self.run_dir / "config.toml"),
                    "--checkpoint",
                    str(self.run_dir / "step_0000000.pt"),
                    "--training-output",
                    str(self.run_dir),
                    "--input-manifest",
                    str(self.val_manifest),
                    "--batch-size",
                    "16",
                    "--output",
                    str(parity),
                ],
                log=self.gate_dir / "parity.log",
            )
        self._evaluate_skyscript(0)
        self._evaluate_rsicd(None)
        self._evaluate_rsicd(0)
        self._evaluate_skyscript(stage)
        self._evaluate_rsicd(stage)

    def _evaluate_skyscript(self, step: int) -> None:
        output = self.gate_dir / f"skyscript_val_step{step:03d}.json"
        if output.exists():
            return
        _run(
            [
                sys.executable,
                "-m",
                "dinotxt_rs.cli.evaluate_skyscript",
                "--config",
                str(self.run_dir / "config.toml"),
                "--manifest",
                str(self.val_manifest),
                "--split",
                "val",
                "--checkpoint",
                str(self.run_dir / f"step_{step:07d}.pt"),
                "--training-output",
                str(self.run_dir),
                "--batch-size",
                str(self.args.batch_size),
                "--num-workers",
                str(self.args.num_workers),
                "--retrieval-chunk-size",
                str(self.args.chunk_size),
                "--output",
                str(output),
            ],
            log=self.gate_dir / f"skyscript_step{step}.log",
        )

    def _evaluate_rsicd(self, step: int | None) -> None:
        suffix = "official" if step is None else f"step{step:03d}"
        output = self.gate_dir / f"rsicd_val_{suffix}.json"
        if output.exists():
            return
        args = [
            sys.executable,
            "-m",
            "dinotxt_rs.cli.evaluate_rsicd",
            "--config",
            str(self.run_dir / "config.toml"),
            "--manifest",
            str(self.rsicd_manifest),
            "--split",
            "val",
        ]
        if step is not None:
            args.extend(
                (
                    "--checkpoint",
                    str(self.run_dir / f"step_{step:07d}.pt"),
                    "--training-output",
                    str(self.run_dir),
                )
            )
        args.extend(
            (
                "--batch-size",
                str(self.args.batch_size),
                "--num-workers",
                str(self.args.num_workers),
                "--retrieval-chunk-size",
                str(self.args.chunk_size),
                "--output",
                str(output),
            )
        )
        _run(args, log=self.gate_dir / f"rsicd_{suffix}.log")

    def run_stage_gate(self, stage: int) -> None:
        training_report = self._ensure_training_report(stage)
        self._ensure_evidence(stage)
        output = self.gate_dir / f"stage_{stage}_summary.json"
        _run(
            [
                sys.executable,
                "tools/summarize_skyscript_gate_m4_b.py",
                "stage",
                "--domain",
                self.domain,
                "--seed",
                str(self.seed),
                "--run-dir",
                str(self.run_dir),
                "--gate-dir",
                str(self.gate_dir),
                "--stage",
                str(stage),
                "--training-report",
                str(training_report),
                "--output",
                str(output),
            ]
        )
        print(f"M4-B {self.domain} seed{self.seed} stage {stage} PASSED.")

    def run(self) -> None:
        self.preflight()
        self.prepare_shared_evidence()
        for stage in STAGES:
            self.train_to_stage(stage)
            self.run_stage_gate(stage)
            if stage == self.args.stop_after_stage:
                if stage == 500:
                    _run(
                        [
                            sys.executable,
                            "tools/summarize_skyscript_gate_m4_b.py",
                            "final",
                            "--domain",
                            self.domain,
                            "--seed",
                            str(self.seed),
                            "--gate-dir",
                            str(self.gate_dir),
                            "--output",
                            str(self.gate_dir / "summary.json"),
                        ]
                    )
                return


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.chunk_size <= 0:
        raise ValueError("Batch/chunk size must be positive and workers nonnegative")
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    M4BRun(args, root).run()


if __name__ == "__main__":
    main()
