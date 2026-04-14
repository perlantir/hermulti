"""Integration test for multi-agent compile routing.

Spawns three heterogeneous persistent-delegate tasks through
:meth:`PersistentDelegateTool.invoke_batch` and asserts:

1. Only ONE compile call reaches HIPP0 (parent-compile-once).
2. Each subagent's slice contains the decision whose content most
   overlaps its task keywords.
3. User facts propagate to every subagent.

Uses the real mock HIPP0 server + stub runner so no LLM is invoked.
"""

from __future__ import annotations

import pytest

from agent.hipp0_memory_provider import Hipp0MemoryProvider
from hermes_cli.agent_registry import AgentConfig, register_agent
from tests.fixtures.mock_hipp0 import start_mock_hipp0
from tools.persistent_delegate_tool import (
    DelegateRunContext,
    DelegateRunResult,
    PersistentDelegateTool,
    _compile_cache_clear,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def _provider_factory(base_url: str, api_key: str):
    def _factory(profile):
        return Hipp0MemoryProvider(
            base_url=base_url,
            api_key=api_key,
            project_id=str(profile.config.project_id),
            agent_name=profile.name,
            agent_id=str(profile.config.agent_id or ""),
            pending_wal_path=profile.pending_wal_path,
            memory_md_path=profile.memory_path,
        )

    return _factory


async def _stub_runner(task, system_prompt, ctx: DelegateRunContext):
    return DelegateRunResult(
        final_message=f"[{ctx.agent.name}] got task: {task}",
        transcript=f"SYSTEM:\n{system_prompt}\n\nUSER: {task}\nASSISTANT: done",
    )


def _seed_three_agents(project_id: str = "proj-shared") -> None:
    for name in ("alice", "bob", "carol"):
        register_agent(
            name,
            soul=f"# {name.title()}\nYou are {name}.\n",
            config=AgentConfig(
                model="anthropic/claude-opus-4.6",
                platform_access=["cli"],
                project_id=project_id,
                agent_id=f"agent-{name}",
            ),
        )


class TestMultiAgentRouting:
    async def test_invoke_batch_compiles_once_and_slices(self, tmp_path):
        _compile_cache_clear()
        _seed_three_agents()

        # Broad compile response: three decisions, each skewed toward
        # one subagent's task, plus a project-wide user fact.
        compile_response = {
            "decisions": [
                {
                    "id": "d-sales",
                    "text": "Enterprise sales kickoff requires stakeholder call",
                    "score": 0.9,
                },
                {
                    "id": "d-bug",
                    "text": "Login crash traceback fix lives in auth handler",
                    "score": 0.88,
                },
                {
                    "id": "d-pref",
                    "text": "User prefers dark mode and terse replies style",
                    "score": 0.85,
                },
            ],
            "total_tokens": 120,
            "cache_hit": False,
            "user_facts": [
                {"key": "timezone", "value": "UTC+1"},
            ],
        }

        async with start_mock_hipp0() as hipp0:
            hipp0.compile_response = compile_response

            captured: list[DelegateRunContext] = []

            async def _spy_runner(task, system_prompt, ctx):
                captured.append(ctx)
                return await _stub_runner(task, system_prompt, ctx)

            tool = PersistentDelegateTool(
                base_url=hipp0.base_url,
                api_key="test-key",
                runner=_spy_runner,
                provider_factory=_provider_factory(hipp0.base_url, "test-key"),
            )

            tasks = [
                {"agent_name": "alice", "task": "draft a sales kickoff email for enterprise"},
                {"agent_name": "bob", "task": "fix the login crash traceback in auth"},
                {"agent_name": "carol", "task": "remember my style preferences for replies"},
            ]

            results = await tool.invoke_batch(tasks, platform="cli")

            assert len(results) == 3

            # (1) Exactly ONE compile call against HIPP0.
            compile_calls = hipp0.calls_for("/api/compile")
            assert len(compile_calls) == 1, (
                f"expected 1 compile call, got {len(compile_calls)}"
            )

            # (2) Each subagent got a task-relevant slice.
            by_agent = {ctx.agent.name: ctx for ctx in captured}
            assert set(by_agent.keys()) == {"alice", "bob", "carol"}

            alice_ids = {d["id"] for d in by_agent["alice"].compiled.decisions}
            bob_ids = {d["id"] for d in by_agent["bob"].compiled.decisions}
            carol_ids = {d["id"] for d in by_agent["carol"].compiled.decisions}

            assert "d-sales" in alice_ids, f"alice: {alice_ids}"
            assert "d-bug" in bob_ids, f"bob: {bob_ids}"
            assert "d-pref" in carol_ids, f"carol: {carol_ids}"

            # Slices are disjoint (each decision goes to exactly one agent).
            assert alice_ids.isdisjoint(bob_ids)
            assert bob_ids.isdisjoint(carol_ids)
            assert alice_ids.isdisjoint(carol_ids)

            # (3) User facts propagate to every subagent.
            for name in ("alice", "bob", "carol"):
                facts = by_agent[name].compiled.user_facts
                assert any(f.get("key") == "timezone" for f in facts), (
                    f"{name} missing user_facts"
                )

    async def test_ttl_cache_absorbs_repeat_compile(self, tmp_path):
        """When invoke() is called twice for the same task, second hits cache."""
        _compile_cache_clear()
        _seed_three_agents()

        async with start_mock_hipp0() as hipp0:
            tool = PersistentDelegateTool(
                base_url=hipp0.base_url,
                api_key="test-key",
                runner=_stub_runner,
                provider_factory=_provider_factory(hipp0.base_url, "test-key"),
            )

            await tool.invoke("alice", "draft a Q3 report")
            await tool.invoke("alice", "draft a Q3 report")

            compile_calls = hipp0.calls_for("/api/compile")
            assert len(compile_calls) == 1, (
                f"expected cache hit on 2nd call; got {len(compile_calls)} compiles"
            )
