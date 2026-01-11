"""Entry point for python -m rex."""

import sys


def main():
    from rex.cli import main as cli_main

    sys.exit(cli_main())


if __name__ == "__main__":
    main()
