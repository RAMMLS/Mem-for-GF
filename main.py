from pathlib import Path

from mem_for_gf.app import cli


if __name__ == "__main__":
    raise SystemExit(cli(Path(__file__).resolve().parent))

