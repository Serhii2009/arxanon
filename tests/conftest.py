"""Shared pytest fixtures for all test modules."""
import pytest

from arxanon import config


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Redirect all data I/O to a temporary directory for every test."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "papers.db")
    monkeypatch.setattr(config, "FAISS_PATH", tmp_path / "embeddings.faiss")
