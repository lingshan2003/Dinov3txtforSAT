import json

from tools.prepare_global77_manifest import derive_records, first_complete_sentence


class FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


def test_first_complete_sentence_does_not_split_at_comma() -> None:
    caption = "Forest, water, and crop are visible. A second sentence follows."
    assert first_complete_sentence(caption) == "Forest, water, and crop are visible."


def test_derive_records_uses_complete_word_backoff(tmp_path) -> None:
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "sample-1",
                "image": "/images/one.png",
                "caption": "one two three four five six. Later details.",
                "split": "train",
                "source": "ChatEarthNet",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records, audit = derive_records(manifest, FakeTokenizer(), context_length=5)

    assert records[0]["caption"] == "one two three"
    assert audit["word_backoff_records"] == 1
    assert audit["global77_caption_tokens"]["max"] == 5


def test_complete_word_strategy_keeps_later_sentences_when_they_fit(tmp_path) -> None:
    manifest = tmp_path / "source.jsonl"
    caption = "An aerial image. It shows: Warehouse."
    manifest.write_text(
        json.dumps(
            {
                "id": "skyscript:sample-1",
                "image": "/images/one.png",
                "caption": caption,
                "split": "train",
                "source": "SkyScript",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records, audit = derive_records(
        manifest,
        FakeTokenizer(),
        context_length=20,
        strategy="complete-word-backoff",
    )

    assert records[0]["caption"] == caption
    assert audit["strategy"] == "complete_word_backoff_without_sentence_selection"
