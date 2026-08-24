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
    COMPLETE_EGRESS_BACKENDS,
    EGRESS_CAPABLE_BACKENDS,
    TCP_ONLY_EGRESS_BACKENDS,
    backend_can_enforce_egress_rules,
    backend_egress_is_tcp_only,
)

GATE_MODULES = (loader, parser, validator)


@pytest.mark.parametrize("backend", ["linux_bwrap", "darwin_seatbelt", "linux_landlock"])
def test_sole_egress_backends_accepted(backend: str) -> None:
    assert backend_can_enforce_egress_rules(backend)


@pytest.mark.parametrize("backend", ["none", "windows_jobobject", "auto", None, ""])
def test_backends_that_cannot_enforce_are_refused(backend: str | None) -> None:
    """Default-deny: anything not proven to hard-enforce is refused."""
    assert not backend_can_enforce_egress_rules(backend)


def test_landlock_can_enforce_but_is_tcp_only() -> None:
    """The distinction a whole-branch review found missing.

    Landlock restricts TCP bind/connect and nothing else -- no UDP, no raw
    sockets, no other families, and not a socket connected before
    restriction. Calling that "the proxy is the only egress path", as an
    earlier version of this predicate did, accepted specs whose declared
    allow-list could be bypassed over a non-TCP transport.
    """
    assert backend_can_enforce_egress_rules("linux_landlock")
    assert backend_egress_is_tcp_only("linux_landlock")


def test_complete_backends_are_not_tcp_only() -> None:
    """bwrap and seatbelt remove the network stack; their guarantee is
    strictly stronger and must not be conflated with Landlock's."""
    for backend in COMPLETE_EGRESS_BACKENDS:
        assert backend_can_enforce_egress_rules(backend)
        assert not backend_egress_is_tcp_only(backend), backend


def test_the_two_sets_are_disjoint_and_cover_the_capable_set() -> None:
    """No backend may be both, and none may go unclassified -- that is how
    a weaker guarantee gets described as a stronger one."""
    assert not (COMPLETE_EGRESS_BACKENDS & TCP_ONLY_EGRESS_BACKENDS)
    assert COMPLETE_EGRESS_BACKENDS | TCP_ONLY_EGRESS_BACKENDS == EGRESS_CAPABLE_BACKENDS


@pytest.mark.parametrize("module", GATE_MODULES, ids=lambda m: m.__name__)
def test_every_gate_site_uses_the_shared_predicate(module: object) -> None:
    """
    Anti-drift. Each module must call the predicate rather than compare the
    backend name itself -- five independent copies of one rule is how the
    list gets widened in some places and not others.
    """
    src = inspect.getsource(module)
    assert "backend_can_enforce_egress_rules" in src, (
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
