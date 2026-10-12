
from pathlib import Path


def test_project_config_files_exist():
    assert Path("config/config.yaml").exists()
    assert Path("params.yaml").exists()
