"""Command-line entry point. Commands are added in later milestones (M2+)."""

from __future__ import annotations

import typer

from automa import __version__

app = typer.Typer(help="Automa command-line interface.", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """Automa command-line interface."""


@app.command()
def version() -> None:
    """Print the installed Automa version."""
    typer.echo(__version__)


def main() -> None:
    """Console-script entry point."""
    app()
