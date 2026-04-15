"""
LLMClient adapter that bridges the SkillDispatcher's minimal Protocol to
hermulti's existing auxiliary_client primitives.

The adapter prefers async clients. If only a sync `call_llm` is available,
it offloads to a thread executor so the turn loop is never blocked.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AuxiliaryLLMAdapter:
    """Adapter that exposes the LLMClient.call(system, user, ...) shape."""

    def __init__(self) -> None:
        self._async_client: Any = None
        self._sync_callable: Any = None
        self._init_clients()

    def _init_clients(self) -> None:
        """Try async clients first, fall back to sync call_llm."""
        try:
            from agent.auxiliary_client import (
                AsyncCodexAuxiliaryClient,
                AsyncAnthropicAuxiliaryClient,
            )
            preferred = os.environ.get('HIPP0_SKILL_LLM_PROVIDER', '').lower()
            if preferred == 'anthropic':
                try:
                    self._async_client = AsyncAnthropicAuxiliaryClient()
                    return
                except Exception as exc:
                    logger.debug('[skill-llm] async anthropic init failed: %s', exc)
            elif preferred in ('codex', 'openai-codex'):
                try:
                    self._async_client = AsyncCodexAuxiliaryClient()
                    return
                except Exception as exc:
                    logger.debug('[skill-llm] async codex init failed: %s', exc)
            else:
                for cls in (AsyncCodexAuxiliaryClient, AsyncAnthropicAuxiliaryClient):
                    try:
                        self._async_client = cls()
                        return
                    except Exception:
                        continue
        except ImportError as exc:
            logger.debug('[skill-llm] async clients unavailable: %s', exc)

        try:
            from agent.auxiliary_client import call_llm
            self._sync_callable = call_llm
        except ImportError as exc:
            logger.debug('[skill-llm] sync call_llm unavailable: %s', exc)

    @property
    def available(self) -> bool:
        return self._async_client is not None or self._sync_callable is not None

    async def call(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> str:
        if self._sync_callable is not None:
            loop = asyncio.get_running_loop()
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

            def _invoke() -> str:
                try:
                    res = self._sync_callable(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except TypeError:
                    res = self._sync_callable(messages)
                if isinstance(res, str):
                    return res
                if isinstance(res, dict):
                    if 'content' in res:
                        return str(res['content'])
                    if 'choices' in res and res['choices']:
                        msg = res['choices'][0].get('message', {})
                        return str(msg.get('content', ''))
                return str(res)

            return await loop.run_in_executor(None, _invoke)

        if self._async_client is not None:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            for method_name in ('call', 'chat', 'complete', 'request'):
                fn = getattr(self._async_client, method_name, None)
                if callable(fn):
                    try:
                        res = await fn(
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        if isinstance(res, str):
                            return res
                        if isinstance(res, dict) and 'content' in res:
                            return str(res['content'])
                    except TypeError:
                        try:
                            res = await fn(messages)
                            if isinstance(res, str):
                                return res
                            if isinstance(res, dict) and 'content' in res:
                                return str(res['content'])
                        except Exception:
                            continue
                    except Exception:
                        continue
            raise RuntimeError('no compatible method found on async client')

        raise RuntimeError('no LLM client available')


def build_skill_llm_client() -> Optional['AuxiliaryLLMAdapter']:
    """Return an adapter if any auxiliary LLM is configured, else None."""
    adapter = AuxiliaryLLMAdapter()
    return adapter if adapter.available else None
