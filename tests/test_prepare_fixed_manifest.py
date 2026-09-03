import json

import pytest

from tools.prepare_fixed_manifest import prepare_fixed_manifest


def _record(index: int, split: str = "train") -> dict[str, str]:
    return {
        "id": f"chatearthnet:sample-{index}",
        "image": f"/images/sample-{index}.png",
        "caption": f"sample {index}",
        "split": split,
        "source": "ChatEarthNet",
    }


def test_prepare_fixed_manifest_preserves_first_records_and_writes_audit(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps(_record(index)) + "\n" for index in range(3)), encoding="utf-8"
    )
    output = tmp_path / "fixed.jsonl"
    audit_output = tmp_path / "fixed.audit.json"

    audit = prepare_fixed_manifest(
        source=source,
        output=output,
        audit_output=audit_output,
        limit=2,
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["id"] for record in records] == [
        "chatearthnet:sample-0",
        "chatearthnet:sample-1",
    ]
    assert audit["selection"] == "first_n_in_existing_manifest_order"
    assert audit["source_records"] == 3
    assert audit["selected_records"] == 2
    assert json.loads(audit_output.read_text(encoding="utf-8")) == audit


def test_prepare_fixed_manifest_rejects_non_training_records(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_record(0, split="val")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="split='train'"):
        prepare_fixed_manifest(
            source=source,
            output=tmp_path / "fixed.jsonl",
            audit_output=tmp_path / "fixed.audit.json",
            limit=1,
        )
