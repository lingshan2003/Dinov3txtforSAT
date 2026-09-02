from tools.prepare_chatearthnet import caption_text


def test_caption_text_accepts_official_singleton_list() -> None:
    assert caption_text(["  A satellite   image. "]) == "A satellite image."

