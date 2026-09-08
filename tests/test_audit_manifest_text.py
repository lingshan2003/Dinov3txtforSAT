import json

from tools.audit_manifest_text import audit_manifest


class WhitespaceTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


def test_audit_reports_exact_duplicates_and_context_overflow(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {"id": "one", "caption": "An aerial image. It shows: Warehouse."},
        {"id": "two", "caption": "An aerial image. It shows: Warehouse."},
        {"id": "three", "caption": "one two three four five six seven"},
    ]
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    report = audit_manifest(manifest, WhitespaceTokenizer(), context_length=8)

    assert report["records"] == 3
    assert report["unique_normalized_captions"] == 2
    assert report["duplicate_records"] == 1
    assert report["overflow_count"] == 1
    assert report["status"] == "context_overflow"
