import argparse
import os
import pathlib
import subprocess
import sys
from typing import Dict

TEMPLATES: Dict[str, str] = {
    "fastapi": """from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/')\ndef read_root():\n    return {'hello': 'world'}\n""",
    "streamlit": """import streamlit as st\nst.title('Hello from ShipFast template')\n""",
}


def write_file(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def scaffold(template: str, target: pathlib.Path) -> None:
    if template not in TEMPLATES:
        raise SystemExit(f"Unknown template: {template}")
    write_file(target, TEMPLATES[template])


def run_uvicorn(app_path: str) -> None:
    subprocess.run([sys.executable, "-m", "uvicorn", app_path, "--reload"], check=False)


def run_streamlit(app_path: str) -> None:
    subprocess.run(["streamlit", "run", app_path], check=False)


def main():
    parser = argparse.ArgumentParser(description="ShipFast CLI - zero-config deploy helper")
    sub = parser.add_subparsers(dest="command")

    init_cmd = sub.add_parser("init", help="Create a scaffolded app")
    init_cmd.add_argument("--template", default="fastapi", choices=list(TEMPLATES.keys()))
    init_cmd.add_argument("--out", default="app.py")

    run_cmd = sub.add_parser("run", help="Run the scaffolded app locally")
    run_cmd.add_argument("--type", default="fastapi", choices=["fastapi", "streamlit"])
    run_cmd.add_argument("--path", default="app.py")

    args = parser.parse_args()

    if args.command == "init":
        target = pathlib.Path(args.out)
        scaffold(args.template, target)
        print(f"Scaffolded {args.template} app at {target}")
        return

    if args.command == "run":
        if args.type == "fastapi":
            run_uvicorn(f"{os.path.splitext(args.path)[0]}:app")
        else:
            run_streamlit(args.path)
        return

    parser.print_help()


if __name__ == "__main__":
    main()


