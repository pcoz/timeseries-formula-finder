"""
Entry point for `python -m ppf`.

Enables running PPF commands directly:
    python -m ppf discover data.csv
    python -m ppf --help
"""

from ppf.cli.main import main

if __name__ == "__main__":
    main()
