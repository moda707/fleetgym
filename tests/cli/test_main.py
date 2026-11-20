import pytest

from fleetgym._version import __version__
from fleetgym.cli.__main__ import main


def test_version_flag(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]):
    assert main([]) == 0
    assert "usage: fleetgym" in capsys.readouterr().out
