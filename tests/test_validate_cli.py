import json
from pathlib import Path

from PIL import Image

from visionops.data.validate_cli import main


def test_validate_cli_writes_json_report(tmp_path: Path, capsys):
    class_dir = tmp_path / "class_a"
    class_dir.mkdir()
    Image.new("RGB", (4, 4), color="black").save(class_dir / "sample.png")
    output = tmp_path / "report.json"

    assert main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) ["valid"] is True
    assert '"total_images": 1' in capsys.readouterr().out
