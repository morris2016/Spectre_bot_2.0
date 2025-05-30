import os
import json
import pytest
from config import Config, load_config
from common.exceptions import ConfigurationError
from common.constants import CONFIG_SCHEMA_VERSION


def test_load_config_defaults(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, Config)
    cfg.validate()



def test_load_config_no_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config("config.json")
    assert isinstance(cfg, Config)
    assert os.path.isfile(tmp_path / "config.json")



def test_validate_missing_db_host():
    cfg = Config()
    cfg.data['database']['host'] = ''
    with pytest.raises(ConfigurationError):
        cfg.validate()



def test_load_config_incorrect_schema_version(tmp_path):
    cfg_file = tmp_path / "config.json"
    data = Config().data
    data["schema_version"] = CONFIG_SCHEMA_VERSION + 1
    with open(cfg_file, "w") as f:
        json.dump(data, f)
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, Config)
    with pytest.raises(ConfigurationError):
        cfg.validate()


@pytest.mark.parametrize("path", ["database.database", "redis.host", "system.name"])
def test_validate_missing_required_fields(path):
    cfg = Config()
    section, key = path.split('.')
    cfg.data[section][key] = ''
    with pytest.raises(ConfigurationError):
        cfg.validate()
