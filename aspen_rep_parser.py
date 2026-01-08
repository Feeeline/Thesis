"""Parse Aspen Plus report (.rep) files into structured data.

This module focuses only on parsing and data extraction. It does not perform
any exergy calculations.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?")


@dataclass
class Environment:
    T0: Optional[float] = None
    P0: Optional[float] = None
    T0_unit: Optional[str] = None
    P0_unit: Optional[str] = None
    reference_composition: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "T0": self.T0,
            "P0": self.P0,
            "T0_unit": self.T0_unit,
            "P0_unit": self.P0_unit,
            "reference_composition": dict(self.reference_composition),
        }


@dataclass
class Stream:
    name: str
    T: Optional[float] = None
    P: Optional[float] = None
    mass_flow: Optional[float] = None
    mol_flow: Optional[float] = None
    enthalpy: Optional[float] = None
    entropy: Optional[float] = None
    density: Optional[float] = None
    phase: Optional[str] = None
    composition_mass: dict[str, float] = field(default_factory=dict)
    composition_mole: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "T": self.T,
            "P": self.P,
            "mass_flow": self.mass_flow,
            "mol_flow": self.mol_flow,
            "enthalpy": self.enthalpy,
            "entropy": self.entropy,
            "density": self.density,
            "phase": self.phase,
            "composition_mass": dict(self.composition_mass),
            "composition_mole": dict(self.composition_mole),
        }


@dataclass
class Block:
    name: str
    block_type: str
    mode: Optional[str] = None

    def to_dict(self) -> dict:
        return {"name": self.name, "type": self.block_type, "mode": self.mode}


@dataclass
class ParsedCase:
    environment: Environment
    streams: list[Stream]
    blocks: list[Block]

    def to_dict(self) -> dict:
        return {
            "environment": self.environment.to_dict(),
            "streams": [stream.to_dict() for stream in self.streams],
            "blocks": [block.to_dict() for block in self.blocks],
        }


SECTION_PATTERNS = {
    "environment": [r"\bAMBIENT\b", r"\bENVIRONMENT\b", r"\bREFERENCE\s+ENVIRONMENT\b"],
    "streams": [r"\bSTREAM\s+REPORT\b", r"\bSTREAM\s+SUMMARY\b"],
    "blocks": [r"\bBLOCK\s+REPORT\b", r"\bBLOCK\s+SUMMARY\b"],
    "mole_frac": [r"\bMOLE\s+FRAC", r"\bMOLE\s+FRACTION"],
    "mass_frac": [r"\bMASS\s+FRAC", r"\bMASS\s+FRACTION"],
}

STREAM_COLUMN_MAP = {
    "TEMP": "T",
    "TEMPERATURE": "T",
    "T": "T",
    "PRES": "P",
    "PRESSURE": "P",
    "P": "P",
    "MASSFLOW": "mass_flow",
    "MASS FLOW": "mass_flow",
    "MASSFL": "mass_flow",
    "MOLEFLOW": "mol_flow",
    "MOLE FLOW": "mol_flow",
    "MOLARFLOW": "mol_flow",
    "ENTHALPY": "enthalpy",
    "H": "enthalpy",
    "ENTROPY": "entropy",
    "S": "entropy",
    "DENSITY": "density",
    "RHO": "density",
    "PHASE": "phase",
}

BLOCK_TYPE_MAP = {
    "VALVE": "Valve",
    "FLASH2": "Flash2",
    "RADFRAC": "Radfrac",
    "COMPR": "COMPR",
    "MIXER": "Mixer",
    "HEATER": "Heater",
    "HEATX": "Heater",
    "MHEATX": "MHeatX",
    "FSPLIT": "FSplit",
}


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().upper())


def _parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    match = NUMBER_RE.search(value.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _find_section_indices(lines: list[str], patterns: Iterable[str]) -> list[int]:
    indices = []
    for idx, line in enumerate(lines):
        for pattern in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                indices.append(idx)
                break
    return indices


def _guess_section_bounds(lines: list[str], start_idx: int) -> tuple[int, int]:
    for idx in range(start_idx + 1, len(lines)):
        if re.match(r"^[A-Z0-9][A-Z0-9\s\-_/]{6,}$", lines[idx].strip()):
            return start_idx, idx
    return start_idx, len(lines)


def _parse_fixed_width_table(lines: list[str], header_idx: int) -> tuple[list[str], list[list[str]], int]:
    if header_idx + 1 >= len(lines):
        return [], [], header_idx
    separator = lines[header_idx + 1]
    spans = [(m.start(), m.end()) for m in re.finditer(r"-{3,}", separator)]
    if len(spans) < 2:
        return [], [], header_idx
    headers = [lines[header_idx][start:end].strip() for start, end in spans]
    rows = []
    idx = header_idx + 2
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            break
        if re.match(r"^[A-Z0-9][A-Z0-9\s\-_/]{6,}$", line.strip()):
            break
        row = [line[start:end].strip() for start, end in spans]
        if any(cell for cell in row):
            rows.append(row)
        idx += 1
    return headers, rows, idx


def _parse_environment(lines: list[str]) -> Environment:
    environment = Environment()
    found = False
    for line in lines:
        if "T0" in line or "P0" in line:
            t0_match = re.search(r"\bT0\b\s*[:=]?\s*([^\s]+)\s*([A-Za-z/]+)?", line)
            if t0_match and environment.T0 is None:
                environment.T0 = _parse_float(t0_match.group(1))
                environment.T0_unit = t0_match.group(2)
                found = True
            p0_match = re.search(r"\bP0\b\s*[:=]?\s*([^\s]+)\s*([A-Za-z/]+)?", line)
            if p0_match and environment.P0 is None:
                environment.P0 = _parse_float(p0_match.group(1))
                environment.P0_unit = p0_match.group(2)
                found = True

    ref_section_indices = _find_section_indices(lines, [r"REFERENCE\s+COMPOSITION", r"REFERENCE\s+COMP"])
    if ref_section_indices:
        start = ref_section_indices[0]
        _, end = _guess_section_bounds(lines, start)
        for line in lines[start:end]:
            if not line.strip() or re.search(r"REFERENCE\s+COMP", line, re.IGNORECASE):
                continue
            parts = line.split()
            if len(parts) >= 2:
                value = _parse_float(parts[-1])
                if value is not None:
                    environment.reference_composition[parts[0]] = value

    if not found and not environment.reference_composition:
        raise ValueError("Environment section not found (T0/P0 or reference composition).")
    return environment


def _parse_streams(lines: list[str]) -> list[Stream]:
    streams: dict[str, Stream] = {}

    for idx, line in enumerate(lines[:-1]):
        if "STREAM" not in line.upper():
            continue
        headers, rows, _ = _parse_fixed_width_table(lines, idx)
        if not headers or not rows:
            continue
        normalized_headers = [_normalize_header(header) for header in headers]
        if not normalized_headers or "STREAM" not in normalized_headers[0]:
            continue
        for row in rows:
            if not row or not row[0].strip():
                continue
            stream_name = row[0].strip()
            stream = streams.setdefault(stream_name, Stream(name=stream_name))
            for header, value in zip(normalized_headers[1:], row[1:]):
                key = STREAM_COLUMN_MAP.get(header)
                if not key:
                    continue
                if key == "phase":
                    if value:
                        stream.phase = value.strip()
                    continue
                parsed = _parse_float(value)
                if parsed is not None:
                    setattr(stream, key, parsed)

    if not streams:
        raise ValueError("Stream section not found (no stream tables detected).")
    return list(streams.values())


def _parse_composition_tables(lines: list[str], streams: dict[str, Stream]) -> None:
    for idx, line in enumerate(lines[:-1]):
        is_mole = bool(re.search(r"MOLE\s+FRAC", line, re.IGNORECASE))
        is_mass = bool(re.search(r"MASS\s+FRAC", line, re.IGNORECASE))
        if not (is_mole or is_mass):
            continue
        for table_idx in range(idx + 1, min(idx + 15, len(lines) - 1)):
            headers, rows, _ = _parse_fixed_width_table(lines, table_idx)
            if not headers or not rows:
                continue
            normalized_headers = [_normalize_header(header) for header in headers]
            if "COMP" not in normalized_headers[0]:
                continue
            stream_names = [header for header in headers[1:] if header.strip()]
            for row in rows:
                component = row[0].strip()
                if not component:
                    continue
                for stream_name, value in zip(stream_names, row[1:]):
                    stream = streams.setdefault(stream_name, Stream(name=stream_name))
                    parsed = _parse_float(value)
                    if parsed is None:
                        continue
                    if is_mole:
                        stream.composition_mole[component] = parsed
                    else:
                        stream.composition_mass[component] = parsed


def _parse_blocks(lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    for idx, line in enumerate(lines[:-1]):
        if "BLOCK" not in line.upper():
            continue
        headers, rows, _ = _parse_fixed_width_table(lines, idx)
        if not headers or not rows:
            continue
        normalized_headers = [_normalize_header(header) for header in headers]
        if "BLOCK" not in normalized_headers[0]:
            continue
        type_index = None
        model_index = None
        for index, header in enumerate(normalized_headers):
            if "TYPE" in header:
                type_index = index
            if "MODEL" in header:
                model_index = index
        for row in rows:
            name = row[0].strip()
            if not name:
                continue
            raw_type = None
            if type_index is not None and type_index < len(row):
                raw_type = row[type_index].strip()
            if raw_type is None and model_index is not None and model_index < len(row):
                raw_type = row[model_index].strip()
            if not raw_type:
                continue
            normalized_type = _normalize_header(raw_type)
            mapped_type = BLOCK_TYPE_MAP.get(normalized_type, raw_type)
            mode = None
            if "TURB" in normalized_type or "TURB" in name.upper():
                mode = "TURBINE"
            elif "COMP" in normalized_type or "COMP" in name.upper():
                mode = "COMPRESSOR"
            blocks.append(Block(name=name, block_type=mapped_type, mode=mode))

    if not blocks:
        raise ValueError("Block section not found (no block tables detected).")
    return blocks


def _parse_bkp_entities(lines: list[str]) -> tuple[list[str], list[str]]:
    stream_names = set()
    block_names = set()
    stream_pattern = re.compile(r"Top\.appModelV8\.Streams\.([A-Za-z0-9_]+)")
    block_pattern = re.compile(r"Top\.appModelV8\.Blocks\.([A-Za-z0-9_]+)")
    for line in lines:
        for match in stream_pattern.finditer(line):
            stream_names.add(match.group(1))
        for match in block_pattern.finditer(line):
            block_names.add(match.group(1))
    return sorted(stream_names), sorted(block_names)


def _parse_lines(lines: list[str]) -> ParsedCase:
    environment = Environment()
    try:
        environment = _parse_environment(lines)
    except ValueError as exc:
        warnings.warn(str(exc), RuntimeWarning)

    streams_list: list[Stream] = []
    try:
        streams_list = _parse_streams(lines)
    except ValueError as exc:
        warnings.warn(str(exc), RuntimeWarning)

    streams = {stream.name: stream for stream in streams_list}
    if streams:
        _parse_composition_tables(lines, streams)
    streams_list = list(streams.values())

    blocks_list: list[Block] = []
    try:
        blocks_list = _parse_blocks(lines)
    except ValueError as exc:
        warnings.warn(str(exc), RuntimeWarning)

    return ParsedCase(environment=environment, streams=streams_list, blocks=blocks_list)


def _build_parse_log(parsed: ParsedCase, source: str, fallback: Optional[str] = None) -> dict:
    stream_logs = []
    for stream in parsed.streams:
        fields = {
            "T": stream.T,
            "P": stream.P,
            "mass_flow": stream.mass_flow,
            "mol_flow": stream.mol_flow,
            "enthalpy": stream.enthalpy,
            "entropy": stream.entropy,
            "density": stream.density,
            "phase": stream.phase,
        }
        units = {
            "T": None,
            "P": None,
            "mass_flow": None,
            "mol_flow": None,
            "enthalpy": None,
            "entropy": None,
            "density": None,
            "phase": None,
        }
        stream_logs.append(
            {
                "name": stream.name,
                "parsed": {key: value for key, value in fields.items() if value is not None},
                "missing": [key for key, value in fields.items() if value is None],
                "units": {key: value for key, value in units.items() if value is not None},
                "units_missing": [key for key, value in units.items() if value is None],
            }
        )

    block_logs = [
        {
            "name": block.name,
            "type": block.block_type,
            "mode": block.mode,
        }
        for block in parsed.blocks
    ]

    log = {
        "source": source,
        "fallback": fallback,
        "environment": {
            "T0": parsed.environment.T0,
            "P0": parsed.environment.P0,
            "T0_unit": parsed.environment.T0_unit,
            "P0_unit": parsed.environment.P0_unit,
            "reference_composition": dict(parsed.environment.reference_composition),
            "units_missing": [
                name
                for name, value in {
                    "T0_unit": parsed.environment.T0_unit,
                    "P0_unit": parsed.environment.P0_unit,
                }.items()
                if value is None
            ],
        },
        "streams": stream_logs,
        "blocks": block_logs,
    }
    return log


def parse_rep_with_log(path: str) -> tuple[ParsedCase, dict]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.rstrip("\n") for line in handle]

    parsed = _parse_lines(lines)
    fallback_used = None
    if input_path.suffix.lower() == ".bkp":
        if not parsed.streams and not parsed.blocks:
            streams, blocks = _parse_bkp_entities(lines)
            if streams or blocks:
                parsed = ParsedCase(
                    environment=parsed.environment,
                    streams=[Stream(name=name) for name in streams],
                    blocks=[Block(name=name, block_type="Unknown") for name in blocks],
                )
        if not parsed.streams and not parsed.blocks:
            rep_candidate = input_path.with_suffix(".rep")
            if rep_candidate.exists():
                with rep_candidate.open("r", encoding="utf-8", errors="ignore") as handle:
                    rep_lines = [line.rstrip("\n") for line in handle]
                parsed = _parse_lines(rep_lines)
                fallback_used = str(rep_candidate)

    log = _build_parse_log(parsed, source=str(input_path), fallback=fallback_used)
    return parsed, log


def parse_rep(path: str) -> ParsedCase:
    parsed, _ = parse_rep_with_log(path)
    return parsed


def _write_json(parsed: ParsedCase, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(parsed.to_dict(), handle, indent=2, ensure_ascii=False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse Aspen Plus report files.")
    parser.add_argument("input", help="Path to Aspen Plus .rep file")
    parser.add_argument("--json", dest="json_path", help="Write parsed output to JSON file")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parsed = parse_rep(args.input)
    if args.json_path:
        _write_json(parsed, args.json_path)
        logging.info("Wrote JSON output to %s", args.json_path)

    env = parsed.environment
    summary = {
        "streams": len(parsed.streams),
        "blocks": len(parsed.blocks),
        "T0": env.T0,
        "P0": env.P0,
    }
    print("Summary:")
    print(f"  Streams: {summary['streams']}")
    print(f"  Blocks: {summary['blocks']}")
    print(f"  T0: {summary['T0']}")
    print(f"  P0: {summary['P0']}")
    print("  Exergies: not parsed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
