"""A registered config is frozen: editing its rule fails this test. A new rule gets a new config."""
from heat import config

FROZEN = {
    "configs/heat_v1.yaml": "b228c97ab677045c2f9e7aa659cc8da16aa881ca308261cb897b2a562729dc8d",  # 2026-10-20
    "configs/heat_v2.yaml": "2097b9dbdc01d1912c15b0c1f1eeacaf483cc139c84a6f5a0c3bb1b2a913baa1",  # 2026-10-23
}


def test_registered_configs_are_unchanged():
    for path, digest in FROZEN.items():
        assert config.digest(path) == digest, f"{path} changed after registration; register a new config"
