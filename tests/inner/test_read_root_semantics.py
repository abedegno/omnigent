"""``with_additional_read_roots`` must not narrow an allow-default-reads backend.

Its semantics were written for deny-default backends, where ``read_roots=None``
means "nothing granted" and adding roots WIDENS the policy. ``linux_landlock``
is allow-default for reads: it handles the read class only when the spec
restricts reads, so ``read_roots=None`` means reads are UNRESTRICTED. Adding
roots there does not widen anything -- it flips reads from unrestricted to an
allow-list containing only the supplied paths.

Found by a real session: claude_sdk_executor adds six internal roots, reads
became restricted to those six, EXECUTE entered the handled mask, and the CLI
could not exec its own binary at /usr/local/bin/claude.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnigent.inner.sandbox import SandboxPolicy, with_additional_read_roots


def _policy(backend_type: str, read_roots: list[Path] | None) -> SandboxPolicy:
    return SandboxPolicy(
        backend_type=backend_type,
        active=True,
        read_roots=read_roots,
        write_roots=[Path("/workspaces")],
        write_files=[],
        allow_network=True,
    )


def test_landlock_unrestricted_reads_are_not_narrowed_by_extra_roots() -> None:
    """The regression. Reads are already unrestricted; you cannot widen
    'everything', and treating None as an empty list denies the rest of the
    filesystem -- including the binary the sandbox exists to launch."""
    policy = _policy("linux_landlock", None)
    out = with_additional_read_roots(policy, [Path("/root/.claude/sessions")])
    assert out.read_roots is None, (
        "adding read roots converted unrestricted reads into an allow-list of "
        f"{out.read_roots!r}; every unlisted path, including the exec target, "
        "is now denied"
    )


def test_landlock_explicit_read_roots_still_accept_extra_roots() -> None:
    """When the spec DID restrict reads, extra grants must still be honoured --
    the fix must not become 'landlock ignores extra roots'."""
    policy = _policy("linux_landlock", [Path("/srv/data")])
    out = with_additional_read_roots(policy, [Path("/root/.claude/sessions")])
    assert out.read_roots is not None
    assert Path("/srv/data") in out.read_roots
    assert Path("/root/.claude/sessions") in out.read_roots


@pytest.mark.parametrize("backend", ["linux_bwrap", "darwin_seatbelt"])
def test_deny_default_backends_keep_their_widening_behaviour(backend: str) -> None:
    """Unchanged for the backends the function was written for: None means
    'no grants', so extra roots must be added rather than dropped."""
    policy = _policy(backend, None)
    out = with_additional_read_roots(policy, [Path("/root/.claude/sessions")])
    assert out.read_roots == [Path("/root/.claude/sessions")]
