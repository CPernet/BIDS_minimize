from __future__ import annotations

import argparse
import copy
import io
import json
import os
import re
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to load the BIDS schema") from exc

BIDS_SCHEMA_JSR_PACKAGE_URL = "https://npm.jsr.io/@jsr/bids__schema"
_GITHUB_SCHEMA_BASE = "https://raw.githubusercontent.com/bids-standard/bids-specification/master/src/schema"
_GITHUB_RAW_RULE_FILES = [
    "anat.yaml",
    "beh.yaml",
    "channels.yaml",
    "dwi.yaml",
    "eeg.yaml",
    "emg.yaml",
    "events.yaml",
    "fmap.yaml",
    "func.yaml",
    "ieeg.yaml",
    "meg.yaml",
    "micr.yaml",
    "motion.yaml",
    "mrs.yaml",
    "nirs.yaml",
    "perf.yaml",
    "pet.yaml",
    "photo.yaml",
    "task.yaml",
]
_MULTI_EXTENSIONS = (".nii.gz", ".tsv.gz")
_BIDS_TOKEN_PATTERN = re.compile(r"^[a-zA-Z0-9]+-.+")


@dataclass
class FileRecord:
    source: Path
    directory: Path
    entities: list[tuple[str, str]]
    suffix: str
    extension: str
    required_entities: set[str]
    optional_entities: list[tuple[str, str]]
    keep_optional_count: int = 0

    def stem(self) -> str:
        kept = []
        optional_kept = 0
        for key, value in self.entities:
            if key in self.required_entities:
                kept.append((key, value))
            elif optional_kept < self.keep_optional_count:
                kept.append((key, value))
                optional_kept += 1
        entity_part = "_".join(f"{k}-{v}" for k, v in kept)
        if entity_part:
            return f"{entity_part}_{self.suffix}"
        return self.suffix

    def destination(self) -> Path:
        return self.directory / f"{self.stem()}{self.extension}"


def _fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "bids-minimize/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def _fetch_text(url: str) -> str:
    return _fetch_bytes(url).decode("utf-8")


def _fetch_json(url: str) -> dict[str, Any]:
    return json.loads(_fetch_text(url))


def _pick_latest_version(version_labels: list[str]) -> str:
    def version_key(label: str) -> tuple[int, ...]:
        core = label.split("-")[0]
        parts = core.split(".")
        normalized: list[int] = []
        for part in parts:
            digits = "".join(ch for ch in part if ch.isdigit())
            normalized.append(int(digits or "0"))
        return tuple(normalized)

    return sorted(version_labels, key=version_key)[-1]


def _load_schema_documents_from_jsr() -> dict[str, dict[str, Any]]:
    registry = _fetch_json(BIDS_SCHEMA_JSR_PACKAGE_URL)
    versions = registry.get("versions") or {}
    if not versions:
        raise RuntimeError("No versions were found in the JSR package metadata")

    latest_version = _pick_latest_version(list(versions.keys()))
    dist = versions[latest_version].get("dist") or {}
    tarball_url = dist.get("tarball")
    if not tarball_url:
        raise RuntimeError("JSR package metadata did not include a tarball URL")

    tarball_bytes = _fetch_bytes(tarball_url)
    docs: dict[str, dict[str, Any]] = {}
    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if not name.endswith(".yaml"):
                continue
            if not (
                name.endswith("src/schema/meta/templates.yaml")
                or name.endswith("src/schema/objects/entities.yaml")
                or "/src/schema/rules/files/raw/" in name
            ):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            docs[Path(name).name] = yaml.safe_load(extracted.read().decode("utf-8"))

    if "templates.yaml" not in docs or "entities.yaml" not in docs:
        raise RuntimeError("Could not locate required schema files in JSR package tarball")

    return docs


def _load_schema_documents_from_github() -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    docs["templates.yaml"] = yaml.safe_load(_fetch_text(f"{_GITHUB_SCHEMA_BASE}/meta/templates.yaml"))
    docs["entities.yaml"] = yaml.safe_load(_fetch_text(f"{_GITHUB_SCHEMA_BASE}/objects/entities.yaml"))
    for rule_file in _GITHUB_RAW_RULE_FILES:
        docs[rule_file] = yaml.safe_load(_fetch_text(f"{_GITHUB_SCHEMA_BASE}/rules/files/raw/{rule_file}"))
    return docs


def load_schema_documents() -> dict[str, dict[str, Any]]:
    try:
        return _load_schema_documents_from_jsr()
    except (
        RuntimeError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        tarfile.TarError,
        yaml.YAMLError,
        KeyError,
        ValueError,
    ):
        return _load_schema_documents_from_github()


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if value is None:
                merged.pop(key, None)
                continue
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _build_schema_tree(docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_rules: dict[str, Any] = {}
    for name, content in docs.items():
        if name in {"templates.yaml", "entities.yaml"}:
            continue
        if isinstance(content, dict):
            raw_rules[name.removesuffix(".yaml")] = content

    return {
        "meta": {"templates": docs["templates.yaml"]},
        "objects": {"entities": docs["entities.yaml"]},
        "rules": {"files": {"raw": raw_rules}},
    }


def _resolve_ref(schema: dict[str, Any], ref: str) -> Any:
    current: Any = schema
    for part in ref.split("."):
        if not isinstance(current, dict) or part not in current:
            return {}
        current = current[part]
    return copy.deepcopy(current)


def _resolve_node(schema: dict[str, Any], node: Any) -> Any:
    if isinstance(node, list):
        return [_resolve_node(schema, item) for item in node]
    if not isinstance(node, dict):
        return copy.deepcopy(node)

    base: dict[str, Any] = {}
    ref_value = node.get("$ref")
    if isinstance(ref_value, str):
        base = _resolve_node(schema, _resolve_ref(schema, ref_value))
    elif isinstance(ref_value, list):
        for ref in reversed(ref_value):
            resolved = _resolve_node(schema, _resolve_ref(schema, ref))
            base = _deep_merge(base, resolved)

    local: dict[str, Any] = {}
    for key, value in node.items():
        if key == "$ref":
            continue
        local[key] = _resolve_node(schema, value)

    return _deep_merge(base, local)


def build_required_entities_by_suffix() -> dict[str, set[str]]:
    docs = load_schema_documents()
    schema = _build_schema_tree(docs)
    resolved_entities = _resolve_node(schema, schema["objects"]["entities"])

    entity_name_map: dict[str, str] = {}
    for canonical_name, entity_def in resolved_entities.items():
        if isinstance(entity_def, dict) and isinstance(entity_def.get("name"), str):
            entity_name_map[canonical_name] = entity_def["name"]
        else:
            entity_name_map[canonical_name] = canonical_name

    required_by_suffix: dict[str, set[str]] = {}
    raw_rules = schema["rules"]["files"]["raw"]
    for _, file_rules in raw_rules.items():
        if not isinstance(file_rules, dict):
            continue
        for _, rule in file_rules.items():
            if not isinstance(rule, dict):
                continue
            resolved_rule = _resolve_node(schema, rule)
            suffixes = resolved_rule.get("suffixes")
            entities = resolved_rule.get("entities")
            if not isinstance(suffixes, list) or not isinstance(entities, dict):
                continue
            required_entities: set[str] = set()
            for entity_key, level in entities.items():
                if isinstance(level, str) and level == "required":
                    required_entities.add(entity_name_map.get(entity_key, entity_key))
                elif isinstance(level, dict) and level.get("level") == "required":
                    required_entities.add(entity_name_map.get(entity_key, entity_key))
            for suffix in suffixes:
                if isinstance(suffix, str):
                    required_by_suffix.setdefault(suffix, set()).update(required_entities)

    return required_by_suffix


def split_extension(name: str) -> tuple[str, str]:
    for ext in _MULTI_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)], ext
    stem, ext = os.path.splitext(name)
    return stem, ext


def parse_bids_filename(name: str) -> tuple[list[tuple[str, str]], str, str] | None:
    stem, extension = split_extension(name)
    if not extension:
        return None

    tokens = stem.split("_")
    if not tokens:
        return None

    suffix = tokens[-1]
    if "-" in suffix:
        return None

    entities: list[tuple[str, str]] = []
    for token in tokens[:-1]:
        if not _BIDS_TOKEN_PATTERN.match(token):
            return None
        key, value = token.split("-", 1)
        entities.append((key, value))

    return entities, suffix, extension


def _collect_records(root: Path, required_by_suffix: dict[str, set[str]]) -> list[FileRecord]:
    records: list[FileRecord] = []
    for dirpath, _, filenames in os.walk(root):
        directory = Path(dirpath)
        for name in filenames:
            parsed = parse_bids_filename(name)
            if parsed is None:
                continue
            entities, suffix, extension = parsed
            required = set(required_by_suffix.get(suffix, set()))
            if not required:
                continue
            optional = [(k, v) for k, v in entities if k not in required]
            records.append(
                FileRecord(
                    source=directory / name,
                    directory=directory,
                    entities=entities,
                    suffix=suffix,
                    extension=extension,
                    required_entities=required,
                    optional_entities=optional,
                )
            )
    return records


def _resolve_collisions(records: list[FileRecord]) -> None:
    while True:
        by_destination: dict[Path, list[FileRecord]] = {}
        for record in records:
            by_destination.setdefault(record.destination(), []).append(record)

        collisions = [group for group in by_destination.values() if len(group) > 1]
        if not collisions:
            return

        for group in collisions:
            max_optional = max(len(record.optional_entities) for record in group)
            depth = 0
            while depth <= max_optional:
                signatures = [tuple(record.optional_entities[:depth]) for record in group]
                if len(set(signatures)) == len(signatures):
                    break
                depth += 1
            for record in group:
                record.keep_optional_count = min(depth, len(record.optional_entities))


def _execute_renames(rename_map: dict[Path, Path]) -> None:
    if not rename_map:
        return

    temp_map: dict[Path, Path] = {}
    for index, (src, dst) in enumerate(rename_map.items()):
        if src == dst:
            continue
        temp_path = src.with_name(f"{src.name}.bidsmin-tmp-{index}")
        os.replace(src, temp_path)
        temp_map[temp_path] = dst

    for temp_path, dst in temp_map.items():
        os.replace(temp_path, dst)


def _update_scans_tsv(root: Path, rename_map: dict[Path, Path], dry_run: bool) -> None:
    if not rename_map:
        return

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith("_scans.tsv"):
                continue
            scans_path = Path(dirpath) / name
            lines = scans_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                continue
            header = lines[0].split("\t")
            if "filename" not in header:
                continue
            idx = header.index("filename")
            changed = False
            updated_lines = [lines[0]]
            for line in lines[1:]:
                cols = line.split("\t")
                if idx >= len(cols):
                    updated_lines.append(line)
                    continue
                rel_filename = cols[idx]
                normalized_rel = os.path.normpath(rel_filename)
                abs_filename = (scans_path.parent / normalized_rel).resolve()
                if abs_filename in rename_map:
                    new_abs = rename_map[abs_filename]
                    cols[idx] = os.path.relpath(new_abs, scans_path.parent).replace(os.sep, "/")
                    changed = True
                updated_lines.append("\t".join(cols))
            if changed and not dry_run:
                scans_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def minimize_bids_filenames(root: str | os.PathLike[str], dry_run: bool = False) -> list[tuple[str, str]]:
    root_path = Path(root).resolve()
    required_by_suffix = build_required_entities_by_suffix()
    records = _collect_records(root_path, required_by_suffix)
    _resolve_collisions(records)

    rename_map: dict[Path, Path] = {}
    for record in records:
        destination = record.destination().resolve()
        source = record.source.resolve()
        if source != destination:
            rename_map[source] = destination

    _update_scans_tsv(root_path, rename_map, dry_run=dry_run)

    if not dry_run:
        _execute_renames(rename_map)

    operations = sorted((str(src), str(dst)) for src, dst in rename_map.items())
    return operations


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimize BIDS filenames to required schema entities")
    parser.add_argument("bids_dir", help="Path to the BIDS dataset")
    parser.add_argument("--dry-run", action="store_true", help="Preview renames without applying them")
    args = parser.parse_args()

    operations = minimize_bids_filenames(args.bids_dir, dry_run=args.dry_run)
    if not operations:
        print("No filenames require minimization.")
        return 0

    for source, destination in operations:
        print(f"{source} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
