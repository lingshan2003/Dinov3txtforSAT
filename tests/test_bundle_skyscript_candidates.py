import csv
import json
import zipfile

import pytest

from tools.bundle_skyscript_candidates import bundle_candidates


def _inputs(tmp_path):
    csv_path = tmp_path / "polished.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filepath", "title_raw", "title"))
        writer.writerows([
            ("images2/a.jpg", "Airport.", "An aerial image. It shows: Airport."),
            ("images2/b.jpg", " Airport. ", "prefix"),
            ("images3/c.jpg", "River.", "prefix"),
            ("images4/d.jpg", "Road.", "prefix"),
        ])
    archives = []
    for name, members in [("images2", [("a", b"original-a"), ("b", b"original-b"),
                                      ("unused", b"unused")]),
                          ("images3", [("c", b"original-c")])]:
        archive = tmp_path / f"{name}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for leaf, content in members:
                handle.writestr(f"{name}/{leaf}.jpg", content)
        archives.append(archive)
    return csv_path, archives


def test_bundle_preserves_all_caption_siblings_and_original_bytes(tmp_path):
    csv_path, archives = _inputs(tmp_path)
    output = tmp_path / "upload.zip"
    report = bundle_candidates(csv_path=csv_path, archives=archives, output=output)
    assert report["candidate_images"] == 3
    assert report["caption_groups"] == 2
    with zipfile.ZipFile(output) as handle:
        assert handle.read("images2/a.jpg") == b"original-a"
        assert handle.read("images2/b.jpg") == b"original-b"
        assert handle.getinfo("images2/a.jpg").compress_type == zipfile.ZIP_STORED
        assert "images2/unused.jpg" not in handle.namelist()
        assert "images4/d.jpg" not in handle.namelist()
        rows = list(csv.DictReader(handle.read("_metadata/candidates.csv").decode().splitlines()))
        assert rows[1]["title_raw"] == " Airport. "
        assert json.loads(handle.read("_metadata/audit.json"))["image_bytes"] == 30
        assert handle.testzip() is None
    assert not output.with_name(output.name + ".part").exists()


def test_dry_run_does_not_extract_or_create_bundle(tmp_path):
    csv_path, archives = _inputs(tmp_path)
    output = tmp_path / "not-created" / "upload.zip"
    report = bundle_candidates(csv_path=csv_path, archives=archives, output=output, dry_run=True)
    assert report["candidate_images"] == 3
    assert not output.parent.exists()
    assert not (tmp_path / "images2").exists()


def test_missing_member_and_existing_output_fail_before_publish(tmp_path):
    csv_path, archives = _inputs(tmp_path)
    output = tmp_path / "upload.zip"
    with pytest.raises(FileNotFoundError, match="not found"):
        bundle_candidates(csv_path=csv_path, archives=archives[:1], output=output)
    assert not output.exists()
    output.write_bytes(b"previous")
    with pytest.raises(FileExistsError):
        bundle_candidates(csv_path=csv_path, archives=archives, output=output)
    assert output.read_bytes() == b"previous"


def test_duplicate_csv_and_wrong_csv_version_rejected(tmp_path):
    csv_path, archives = _inputs(tmp_path)
    output = tmp_path / "upload.zip"
    with csv_path.open("a") as stream:
        stream.write("images2/a.jpg,Airport.,prefix\n")
    with pytest.raises(ValueError, match="Duplicate selected filepath"):
        bundle_candidates(csv_path=csv_path, archives=archives, output=output)
    csv_path.write_text("filepath,title\nimages2/a.jpg,airport\n")
    with pytest.raises(ValueError, match="polished top30"):
        bundle_candidates(csv_path=csv_path, archives=archives, output=output)


def test_failed_stream_does_not_publish_or_modify_sources(tmp_path, monkeypatch):
    csv_path, archives = _inputs(tmp_path)
    original = [archive.read_bytes() for archive in archives]
    output = tmp_path / "upload.zip"

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr("tools.bundle_skyscript_candidates.shutil.copyfileobj", fail)
    with pytest.raises(OSError, match="disk failure"):
        bundle_candidates(csv_path=csv_path, archives=archives, output=output)
    assert not output.exists()
    assert not output.with_name(output.name + ".part").exists()
    assert [archive.read_bytes() for archive in archives] == original
