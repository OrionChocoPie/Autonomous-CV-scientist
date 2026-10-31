import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cv_scientist.cli import main
from cv_scientist.credentials import global_credentials_path, load_global_api_key, save_global_api_key
from cv_scientist.meta import MetaConfig


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("CV_SCIENTIST_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def workspace(root):
    root.mkdir(exist_ok=True)
    return SimpleNamespace(root=str(root), config={})


def test_config_prompts_privately_without_creating_workspace(isolated_config, monkeypatch, capsys):
    prompts = []
    def getpass(prompt):
        prompts.append(prompt)
        return "private-test-key"
    monkeypatch.setattr("getpass.getpass", getpass)
    assert main(["config"]) == 0
    assert prompts
    path = global_credentials_path()
    assert json.loads(path.read_text()) == {"api_key": "private-test-key"}
    assert path.stat().st_mode & 0o777 == 0o600
    output = capsys.readouterr()
    assert "private-test-key" not in output.out + output.err
    assert str(path) in output.out
    assert not (isolated_config / ".cv-scientist").exists()


def test_global_key_is_shared_across_projects_without_mutating_environment(isolated_config, monkeypatch):
    save_global_api_key("shared-test-key")
    for name in ("first", "second"):
        assert MetaConfig.from_workspace(workspace(isolated_config / name)).api_key == "shared-test-key"
    import os
    assert "CV_SCIENTIST_API_KEY" not in os.environ


def test_key_precedence_and_legacy_name_is_ignored(isolated_config, monkeypatch):
    ws = workspace(isolated_config / "project")
    save_global_api_key("global-key")
    monkeypatch.setenv("MODEL_API_KEY", "unrelated-key")
    assert MetaConfig.from_workspace(ws).api_key == "global-key"
    (isolated_config / ".env").write_text("CV_SCIENTIST_API_KEY=cwd-key\n")
    assert MetaConfig.from_workspace(ws).api_key == "cwd-key"
    (Path(ws.root) / ".env").write_text("CV_SCIENTIST_API_KEY=project-key\n")
    assert MetaConfig.from_workspace(ws).api_key == "project-key"
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "environment-key")
    assert MetaConfig.from_workspace(ws).api_key == "environment-key"


def test_missing_global_file_does_not_create_anything(isolated_config):
    assert load_global_api_key() == ""
    assert not global_credentials_path().parent.exists()


def test_legacy_key_alone_cannot_authenticate(isolated_config, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "unrelated-key")
    (isolated_config / ".env").write_text("MODEL_API_KEY=legacy-local-key\n")
    with pytest.raises(ValueError, match="CV_SCIENTIST_API_KEY"):
        MetaConfig.from_workspace(workspace(isolated_config / "project"))


@pytest.mark.parametrize("key", [None, "", "   ", 123])
def test_invalid_key_does_not_overwrite_existing_config(isolated_config, key):
    save_global_api_key("original-key")
    before = global_credentials_path().read_bytes()
    with pytest.raises(ValueError, match="CV_SCIENTIST_API_KEY"):
        save_global_api_key(key)
    assert global_credentials_path().read_bytes() == before


def test_empty_environment_key_is_an_error_instead_of_global_fallback(isolated_config, monkeypatch):
    save_global_api_key("global-key")
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "")
    with pytest.raises(ValueError, match="CV_SCIENTIST_API_KEY"):
        MetaConfig.from_workspace(workspace(isolated_config / "project"))


def test_reconfiguration_replaces_key_and_keeps_private_permissions(isolated_config, monkeypatch, capsys):
    save_global_api_key("old-test-key")
    path = global_credentials_path()
    path.chmod(0o644)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "new-test-key")
    assert main(["config"]) == 0
    assert load_global_api_key() == "new-test-key"
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(path.parent.iterdir()) == [path]
    output = capsys.readouterr()
    assert "test-key" not in output.out + output.err


@pytest.mark.parametrize("content", ['{"api_key": "secret-value",', "[]", "{}", '{"api_key": null}', '{"api_key": " "}'])
def test_invalid_global_config_errors_without_echoing_contents(isolated_config, content):
    path = global_credentials_path()
    path.parent.mkdir(parents=True)
    path.write_text(content)
    with pytest.raises(ValueError) as error:
        load_global_api_key()
    assert "secret-value" not in str(error.value)


def test_environment_override_does_not_require_valid_global_file(isolated_config, monkeypatch):
    path = global_credentials_path()
    path.parent.mkdir(parents=True)
    path.write_text("invalid JSON")
    monkeypatch.setenv("CV_SCIENTIST_API_KEY", "environment-key")
    assert MetaConfig.from_workspace(workspace(isolated_config / "project")).api_key == "environment-key"


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_cancelled_config_does_not_write_key(isolated_config, monkeypatch, capsys, error):
    def cancel(prompt): raise error()
    monkeypatch.setattr("getpass.getpass", cancel)
    assert main(["config"]) != 0
    assert not global_credentials_path().exists()
    assert "cancelled" in capsys.readouterr().err


def test_empty_prompt_is_an_error_without_writing_config(isolated_config, monkeypatch, capsys):
    monkeypatch.setattr("getpass.getpass", lambda prompt: " ")
    assert main(["config"]) == 2
    assert not global_credentials_path().exists()
    assert "CV_SCIENTIST_API_KEY" in capsys.readouterr().err


def test_config_refuses_echoing_fallback_without_terminal(isolated_config, monkeypatch, capsys):
    import getpass
    import warnings
    def no_terminal(prompt): warnings.warn("Cannot hide input", getpass.GetPassWarning)
    monkeypatch.setattr(getpass, "getpass", no_terminal)
    assert main(["config"]) == 2
    assert not global_credentials_path().exists()
    assert "requires a terminal" in capsys.readouterr().err


def test_default_config_location_uses_user_home(isolated_config, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert global_credentials_path() == isolated_config / "home" / ".config" / "cv-scientist" / "credentials.json"


def test_relative_xdg_config_directory_is_an_error(isolated_config, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/config")
    with pytest.raises(ValueError, match="absolute path"): save_global_api_key("test-key")
    assert not (isolated_config / "relative").exists()


def test_failed_atomic_save_preserves_existing_key(isolated_config, monkeypatch):
    save_global_api_key("original-key")
    def fail(self, target): raise OSError("Cannot replace credentials")
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError): save_global_api_key("replacement-key")
    assert load_global_api_key() == "original-key"
    assert list(global_credentials_path().parent.iterdir()) == [global_credentials_path()]
