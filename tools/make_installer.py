"""Build the Windows installer (setup.exe) from the PyInstaller output.

    python tools/make_installer.py --dist dist/noScribe --out installer

Fills pyinstaller/nsis_ma.nsi.in and runs makensis (preinstalled on the
GitHub Windows runners; otherwise install NSIS and put makensis on PATH).
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def app_version() -> str:
    text = (ROOT / "noScribe" / "main.py").read_text(encoding="utf-8")
    return re.search(r"^app_version = '([^']+)'", text, re.M).group(1)


def four_part(version: str) -> str:
    parts = re.findall(r"\d+", version)[:4]
    return ".".join(parts + ["0"] * (4 - len(parts)))


def find_makensis() -> str:
    found = shutil.which("makensis")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base and Path(base, "NSIS", "makensis.exe").exists():
            return str(Path(base, "NSIS", "makensis.exe"))
    raise SystemExit("makensis not found; install NSIS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default="dist/noScribe")
    ap.add_argument("--out", default="installer")
    ap.add_argument("--suffix", default="", help="e.g. -ma1, added to the file name")
    args = ap.parse_args()

    dist = Path(args.dist).resolve()
    if not (dist / "noScribe.exe").exists():
        raise SystemExit(f"No noScribe.exe in {dist}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    version = app_version()
    name = f"noScribe_setup_{version.replace('.', '_')}{args.suffix}.exe"

    script = (ROOT / "pyinstaller" / "nsis_ma.nsi.in").read_text(encoding="utf-8")
    values = {
        "version": four_part(version),
        "year": str(datetime.date.today().year),
        "license": str(ROOT / "LICENSE.txt"),
        "installer_name": str(out / name),
        "source_dir": str(dist),
        "icon": str(ROOT / "img" / "noScribeLogo.ico"),
    }
    for key, value in values.items():
        script = script.replace(f"#*{key}*#", value)
    nsi = out / "noScribe.nsi"
    nsi.write_text(script, encoding="utf-8-sig")  # BOM: NSIS reads it as UTF-8

    subprocess.run([find_makensis(), "/V2", "/INPUTCHARSET", "UTF8", str(nsi)], check=True)
    size = (out / name).stat().st_size / 1e6
    print(f"Installer: {out / name} ({size:,.0f} MB)")


if __name__ == "__main__":
    main()
