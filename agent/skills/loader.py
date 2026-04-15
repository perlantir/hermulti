"""
SkillLoader: parses RESOLVER.md and SKILL.md files from a skills directory
into Python dataclasses.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    name: str
    version: str
    description: str
    triggers: list[str] = field(default_factory=list)
    mutating: bool = False
    tools: list[str] = field(default_factory=list)
    body: str = ""  # Markdown body after frontmatter
    path: str = ""  # Source file path


@dataclass
class ResolverEntry:
    """One row from the RESOLVER.md routing table."""
    trigger_text: str  # raw human text (used for documentation, not matching)
    skill_name: str    # the skill it routes to


@dataclass
class SkillSet:
    skills: list[Skill]
    resolver: list[ResolverEntry]
    skills_dir: str

    def get(self, name: str) -> Optional[Skill]:
        for s in self.skills:
            if s.name == name:
                return s
        return None


# Match a YAML frontmatter block at the top of a markdown file
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
# Parse a markdown table row: | col1 | col2 |
_TABLE_ROW_RE = re.compile(r"^\|\s*(.+?)\s*\|\s*([^|]+?)\s*\|\s*$")


def _parse_yaml_value(raw: str) -> object:
    """Minimal YAML scalar parser sufficient for our SKILL.md frontmatter."""
    raw = raw.strip()
    if raw.startswith('[') and raw.endswith(']'):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [v.strip().strip('"\'') for v in inner.split(',')]
    if raw.lower() in ('true', 'yes'):
        return True
    if raw.lower() in ('false', 'no'):
        return False
    return raw.strip('"\'')


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Return (frontmatter_dict, body)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    yaml_block, body = m.group(1), m.group(2)
    fm: dict[str, object] = {}
    current_list_key: str | None = None
    current_list: list[str] = []

    for raw_line in yaml_block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        # List item: "  - value"
        list_match = re.match(r"^\s*-\s*(.+)$", line)
        if list_match and current_list_key is not None:
            current_list.append(list_match.group(1).strip().strip('"\''))
            continue
        # Key: value (or "key:" introducing a list)
        kv_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
        if kv_match:
            # Flush previous list
            if current_list_key is not None:
                fm[current_list_key] = current_list
                current_list_key = None
                current_list = []
            key = kv_match.group(1)
            val = kv_match.group(2).strip()
            if val == "":
                # Start a list
                current_list_key = key
                current_list = []
            else:
                fm[key] = _parse_yaml_value(val)
    # Flush trailing list
    if current_list_key is not None:
        fm[current_list_key] = current_list

    return fm, body.lstrip("\n")


def _load_skill_file(path: Path) -> Optional[Skill]:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return None
    fm, body = _parse_frontmatter(text)
    if not fm.get('name'):
        return None
    triggers = fm.get('triggers') or []
    tools = fm.get('tools') or []
    return Skill(
        name=str(fm.get('name', '')),
        version=str(fm.get('version', '0.0.0')),
        description=str(fm.get('description', '')),
        triggers=list(triggers) if isinstance(triggers, list) else [],
        mutating=bool(fm.get('mutating', False)),
        tools=list(tools) if isinstance(tools, list) else [],
        body=body,
        path=str(path),
    )


def _parse_resolver(path: Path) -> list[ResolverEntry]:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return []

    entries: list[ResolverEntry] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        # Skip header rows (contain --- separators) and the labels row
        if not line.startswith('|'):
            continue
        if '---' in line:
            continue
        m = _TABLE_ROW_RE.match(line)
        if not m:
            continue
        col1 = m.group(1).strip()
        col2 = m.group(2).strip()
        # Skip the header row
        if col1.lower() == 'trigger' and col2.lower() == 'skill':
            continue
        # Extract skill name from `skill-name` (backticked) or first word
        skill_match = re.search(r'`([a-zA-Z0-9_\-]+)`', col2)
        if skill_match:
            skill_name = skill_match.group(1)
        else:
            skill_name = col2.split()[0] if col2 else ''
        if not skill_name:
            continue
        entries.append(ResolverEntry(trigger_text=col1, skill_name=skill_name))
    return entries


def load_skills(skills_dir: str | None = None) -> SkillSet:
    """Load all skills + the RESOLVER table from the given directory.
    Default: HIPP0_SKILLS_DIR env, falling back to /root/audit/hipp0ai/skills.
    """
    base = Path(skills_dir or os.environ.get('HIPP0_SKILLS_DIR') or '/root/audit/hipp0ai/skills')

    resolver_path = base / 'RESOLVER.md'
    resolver = _parse_resolver(resolver_path) if resolver_path.exists() else []

    skills: list[Skill] = []
    if base.exists():
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            skill_md = sub / 'SKILL.md'
            if not skill_md.exists():
                continue
            sk = _load_skill_file(skill_md)
            if sk:
                skills.append(sk)

    return SkillSet(skills=skills, resolver=resolver, skills_dir=str(base))
