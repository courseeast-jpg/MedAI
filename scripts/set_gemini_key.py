from __future__ import annotations

from getpass import getpass
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    key = getpass("GEMINI_API_KEY: ").strip()

    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    replaced = False
    updated: list[str] = []
    for line in lines:
        if line.startswith("GEMINI_API_KEY="):
            updated.append(f"GEMINI_API_KEY={key}")
            replaced = True
        else:
            updated.append(line)

    if not replaced:
        updated.append(f"GEMINI_API_KEY={key}")

    env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print("GEMINI_API_KEY saved to .env")


if __name__ == "__main__":
    main()
