"""The sole-egress backend gate, and the anti-drift check for its call sites.

Five sites gate egress_rules / credential_proxy on the sandbox backend. They
previously hardcoded the backend list independently and had already begun to
drift -- one used a named constant, four an inline tuple -- which is how a
change that widened only some of them would have left the feature refused.
"""

from __future__ import annotations

import inspect

import pytest

import omnigent.inner.loader as loader
import omnigent.spec.parser as parser
import omnigent.spec.validator as validator
from omnigent.inner.datamodel import (
    SOLE_EGRESS_BACKENDS,
    backend_hard_enforces_sole_egress,
)

GATE_MODULES = (loader, parser, validator)


@pytest.mark.parametrize("backend", ["linux_bwrap", "darwin_seatbelt", "linux_landlock"])
def test_sole_egress_backends_accepted(backend: str) -> None:
    assert backend_hard_enforces_sole_egress(backend)


@pytest.mark.parametrize("backend", ["none", "windows_jobobject", "auto", None, ""])
def test_backends_that_cannot_enforce_are_refused(backend: str | None) -> None:
    """Default-deny: anything not proven to hard-enforce is refused."""
    assert not backend_hard_enforces_sole_egress(backend)


def test_landlock_is_in_the_set() -> None:
    """The change this test exists for. Without landlock the deployment
    that cannot run bwrap has no way to declare egress rules at all."""
    assert "linux_landlock" in SOLE_EGRESS_BACKENDS


@pytest.mark.parametrize("module", GATE_MODULES, ids=lambda m: m.__name__)
def test_every_gate_site_uses_the_shared_predicate(module: object) -> None:
    """
    Anti-drift. Each module must call the predicate rather than compare the
    backend name itself -- five independent copies of one rule is how the
    list gets widened in some places and not others.
    """
    src = inspect.getsource(module)
    assert "backend_hard_enforces_sole_egress" in src, (
        f"{module.__name__} does not use the shared predicate"
    )
    assert 'not in ("linux_bwrap"' not in src, f"{module.__name__} still hardcodes a backend list"


def test_no_module_hardcodes_the_backend_tuple() -> None:
    """Belt and braces across the whole package, not just the three modules
    this test knows about: a sixth gate could be added tomorrow."""
    import pathlib

    root = pathlib.Path(loader.__file__).parent.parent
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if 'sandbox_type not in ("linux_bwrap"' in p.read_text(errors="ignore")
    ]
    assert not offenders, f"hardcoded backend gates remain: {offenders}"
