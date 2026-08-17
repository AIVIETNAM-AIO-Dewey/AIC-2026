from __future__ import annotations

import json

import pytest
from aic2026.object_description.dam_backend import verify_installed_dam_revision


class _FakeDistribution:
    def __init__(self, direct_url: dict[str, object] | None) -> None:
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return json.dumps(self.direct_url) if self.direct_url is not None else None


def test_dam_revision_is_verified_from_pep610(monkeypatch: pytest.MonkeyPatch) -> None:
    revision = "a" * 40
    monkeypatch.setattr(
        "aic2026.object_description.dam_backend.importlib.metadata.distribution",
        lambda _: _FakeDistribution({"vcs_info": {"commit_id": revision}}),
    )

    assert verify_installed_dam_revision(revision) == "pep610"


def test_offline_dam_revision_requires_explicit_verified_mirror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "b" * 40
    monkeypatch.setattr(
        "aic2026.object_description.dam_backend.importlib.metadata.distribution",
        lambda _: _FakeDistribution(None),
    )
    monkeypatch.delenv("AIC_DAM_CODE_REVISION", raising=False)
    with pytest.raises(RuntimeError, match="Cannot verify"):
        verify_installed_dam_revision(revision)

    monkeypatch.setenv("AIC_DAM_CODE_REVISION", revision)
    assert verify_installed_dam_revision(revision) == "offline_mirror_checksum"
