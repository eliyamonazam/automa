"""``python -m automa`` — the same entry point as the ``automa`` command.

Used by the web UI to start ``automa run`` in the background (D125)."""

from automa.cli import main

if __name__ == "__main__":
    main()
