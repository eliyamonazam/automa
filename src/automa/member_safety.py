"""Safe mode for the projects of a ``member`` account (M15, D185).

A member (a family member with no technical background) cannot judge what a
command does to the machine Automa runs on. So whatever ``automa.toml`` says for
the owner's own projects, a member's project always runs with:

* ``[sandbox] mode = "docker"``: every ``run_command`` (and every command check of
  the verifier, which goes through the same tools) runs in a throw-away, isolated
  container without network (D32). When the owner's configuration already uses
  Docker, its settings (image, limits) are kept; otherwise the defaults apply.
  If Docker is not usable nothing runs at all: there is no fallback to the host;
* ``[web] auto_approve_reads = false``: no tool call is ever approved
  automatically; every action that needs the Approval Gate waits for a person.

``member_config`` only ever makes a configuration stricter: no other setting is
touched, and a configuration that is already safe is returned unchanged.
"""

from __future__ import annotations

from automa.config import AutomaConfig, SandboxConfig, SandboxMode


def member_config(config: AutomaConfig) -> AutomaConfig:
    """``config`` with the member safe mode applied (see the module docstring)."""
    update: dict[str, object] = {}
    if config.sandbox.mode is not SandboxMode.DOCKER:
        update["sandbox"] = SandboxConfig(mode=SandboxMode.DOCKER)
    if config.web.auto_approve_reads:
        update["web"] = config.web.model_copy(update={"auto_approve_reads": False})
    if not update:
        return config
    return config.model_copy(update=update)


def is_member_safe(config: AutomaConfig) -> bool:
    """Whether ``config`` satisfies the member safe mode."""
    return config.sandbox.mode is SandboxMode.DOCKER and not config.web.auto_approve_reads
