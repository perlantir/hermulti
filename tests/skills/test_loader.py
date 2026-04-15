"""Tests for the SkillLoader."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent.skills.loader import (
    Skill,
    ResolverEntry,
    load_skills,
    _parse_frontmatter,
    _parse_resolver,
)


def _write_skill_dir(tmp_path: Path, name: str, frontmatter: str, body: str) -> None:
    sub = tmp_path / name
    sub.mkdir()
    (sub / 'SKILL.md').write_text(f"---\n{frontmatter}\n---\n{body}", encoding='utf-8')


def test_parse_frontmatter_basic():
    text = textwrap.dedent("""\
        ---
        name: my-skill
        version: 1.2.3
        description: Test skill
        mutating: true
        ---
        # Body content
        Stuff.
    """)
    fm, body = _parse_frontmatter(text)
    assert fm['name'] == 'my-skill'
    assert fm['version'] == '1.2.3'
    assert fm['description'] == 'Test skill'
    assert fm['mutating'] is True
    assert body.startswith('# Body content')


def test_parse_frontmatter_lists():
    text = textwrap.dedent("""\
        ---
        name: x
        triggers:
          - first trigger
          - second trigger
        tools: [tool_a, tool_b]
        ---
        body
    """)
    fm, _ = _parse_frontmatter(text)
    assert fm['triggers'] == ['first trigger', 'second trigger']
    assert fm['tools'] == ['tool_a', 'tool_b']


def test_load_skills_from_dir(tmp_path: Path):
    # RESOLVER.md
    (tmp_path / 'RESOLVER.md').write_text(textwrap.dedent("""\
        # Resolver

        | Trigger | Skill |
        |---------|-------|
        | Every inbound message | `signal-detector` |
        | Starting a task | `compile-context` |
    """), encoding='utf-8')

    _write_skill_dir(tmp_path, 'signal-detector',
        'name: signal-detector\nversion: 1.0.0\ndescription: Detects signals.\nmutating: true\ntriggers:\n  - every inbound message\ntools: [hipp0_record_decision]',
        '# Signal Detector\nDoes things.')
    _write_skill_dir(tmp_path, 'compile-context',
        'name: compile-context\nversion: 1.0.0\ndescription: Loads context.\nmutating: false\ntriggers:\n  - starting a task',
        '# Compile')

    ss = load_skills(skills_dir=str(tmp_path))
    assert len(ss.skills) == 2
    assert ss.get('signal-detector') is not None
    assert ss.get('signal-detector').mutating is True
    assert 'hipp0_record_decision' in ss.get('signal-detector').tools
    assert len(ss.resolver) == 2
    assert ss.resolver[0].skill_name == 'signal-detector'


def test_load_skills_real_directory():
    """Load the actual hipp0ai skills directory and verify expected skills exist."""
    ss = load_skills(skills_dir='/root/audit/hipp0ai/skills')
    skill_names = {s.name for s in ss.skills}
    # Should include all 8 core skills
    expected = {
        'signal-detector', 'brain-ops', 'compile-context',
        'capture-decision', 'record-outcome', 'search-decisions',
        'maintain', 'synthesize-branch',
    }
    missing = expected - skill_names
    assert not missing, f"Missing skills: {missing}"


def test_resolver_skips_header():
    text = textwrap.dedent("""\
        | Trigger | Skill |
        |---------|-------|
        | Foo     | `bar` |
    """)
    import io
    p = Path('/tmp/_test_resolver.md')
    p.write_text(text, encoding='utf-8')
    entries = _parse_resolver(p)
    p.unlink()
    assert len(entries) == 1
    assert entries[0].skill_name == 'bar'
