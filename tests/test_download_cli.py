from visionops.data.download import build_parser


def test_download_cli_requires_explicit_dataset():
    args = build_parser().parse_args(["--dataset", "eurosat", "--root", "data/raw/eurosat"])
    assert args.download is False
