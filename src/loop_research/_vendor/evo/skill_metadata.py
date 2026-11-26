"""Unmodified metadata helpers extracted from pinned EvoScientist; see UPSTREAM.json."""
from __future__ import annotations
import logging
import re
from pathlib import Path
import yaml
_logger=logging.getLogger(__name__)

_MANIFEST_FILENAME = ".installed.yaml"

_ManifestEntry = dict[str, str]

def _manifest_path(dest_dir: str | Path) -> Path:
    return Path(dest_dir) / _MANIFEST_FILENAME

def _load_manifest(dest_dir: str | Path) -> dict[str, _ManifestEntry]:
    """Read the install manifest from *dest_dir*. Malformed entries are
    skipped, and any read error returns ``{}`` so a corrupt file never breaks
    installation or detection.
    """
    path = _manifest_path(dest_dir)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, RecursionError, yaml.YAMLError) as exc:
        _logger.warning("Failed to read skills manifest at %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, _ManifestEntry] = {}
    for k, v in data.items():
        if not isinstance(v, dict) or not isinstance(v.get("source"), str):
            continue
        entry: _ManifestEntry = {"source": v["source"]}
        commit = v.get("commit")
        if isinstance(commit, str) and commit:
            entry["commit"] = commit
        out[str(k)] = entry
    return out

def installed_skill_provenance(skill_dir: str | Path) -> dict[str, str] | None:
    """Return install provenance recorded for an installed skill directory.

    The sidecar manifest is keyed by the physical directory name, so this
    helper is useful to callers that already resolved the skill path. A
    missing entry means the skill was created locally or predates provenance
    tracking.
    """
    path = Path(skill_dir)
    entry = _load_manifest(path.parent).get(path.name)
    return dict(entry) if entry is not None else None

_MAX_SKILL_MD_BYTES = 10 * 1024 * 1024

_INDEX_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

def _skill_name_and_description(raw: bytes) -> tuple[str, str] | None:
    """Return the frontmatter name and description, or None when the agent skips the file."""
    if len(raw) > _MAX_SKILL_MD_BYTES:
        return None
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    match = _INDEX_FRONTMATTER_RE.match(content)
    if match is None:
        return None
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except (yaml.YAMLError, ValueError, RecursionError):
        # safe_load also raises these for a bad date or tag and for deep nesting.
        return None
    if not isinstance(frontmatter, dict):
        return None
    name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()
    if not name or not description:
        return None
    return name, description
