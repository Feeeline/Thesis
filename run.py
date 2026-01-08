# run_parse_once.py
from __future__ import annotations

from pathlib import Path
import json

# Import aus deinem Parser-File (muss im selben Ordner liegen)
from aspen_rep_parser import parse_rep


# -----------------------------
# HIER DEIN PFAD (Windows-Beispiel)
# -----------------------------
REP_PATH = r"C:\Users\Felin\Documents\Masterthesis\Code\Thesis\Thesis\Aspenfile\Doppelkolonne.bkp"

# Optional: Ausgabeordner + Dateiname
OUT_DIR = Path(r"C:\Users\Felin\Documents\Masterthesis\Code\Thesis\Thesis\Aspenfile")
# Only provide a filename here to avoid Windows backslash escape issues
OUT_JSON_NAME = "Doppelkolonne_parsed.json"


def main() -> None:
    rep_file = Path(REP_PATH)

    if not rep_file.exists():
        raise FileNotFoundError(
            f"REP file not found:\n{rep_file}\n\n"
            "Tipp: Prüfe den Pfad in REP_PATH (oben) und nutze ein raw string r'...'."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / OUT_JSON_NAME

    parsed = parse_rep(str(rep_file))

    # JSON speichern
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(parsed.to_dict(), f, indent=2, ensure_ascii=False)

    # Kurzes Summary in der Konsole
    env = parsed.environment
    print("\n✅ Parsing fertig!")
    print(f"Input:  {rep_file}")
    print(f"Output: {out_json}")
    print("\nSummary:")
    print(f"  Streams: {len(parsed.streams)}")
    print(f"  Blocks:  {len(parsed.blocks)}")
    print(f"  T0:      {env.T0} {env.T0_unit or ''}".rstrip())
    print(f"  P0:      {env.P0} {env.P0_unit or ''}".rstrip())

    # Optional: Liste der Blocktypen
    if parsed.blocks:
        types = sorted({b.block_type for b in parsed.blocks})
        print("\nBlock types found:", ", ".join(types))


if __name__ == "__main__":
    main()
