from pathlib import Path


def test_no_argument_cli_path_is_supported():
    source = Path("ai_fingerprint/cli.py").read_text(encoding="utf-8")
    assert "if len(sys.argv) == 1" in source
    assert "interactive_configure()" in source
    assert "run(config)" in source
