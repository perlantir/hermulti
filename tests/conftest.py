"""Shared fixtures for the hermes-agent test suite."""

import asyncio
import os
import signal
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME to a temp dir so tests never write to ~/.hermes/."""
    fake_home = tmp_path / "hermes_test"
    fake_home.mkdir()
    (fake_home / "sessions").mkdir()
    (fake_home / "cron").mkdir()
    (fake_home / "memories").mkdir()
    (fake_home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    # Reset plugin singleton so tests don't leak plugins from ~/.hermes/plugins/
    try:
        import hermes_cli.plugins as _plugins_mod
        monkeypatch.setattr(_plugins_mod, "_plugin_manager", None)
    except Exception:
        pass
    # Tests should not inherit the agent's current gateway/messaging surface.
    # Individual tests that need gateway behavior set these explicitly.
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    # Avoid making real calls during tests if this key is set in the env files
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture()
def tmp_dir(tmp_path):
    """Provide a temporary directory that is cleaned up automatically."""
    return tmp_path


@pytest.fixture()
def mock_config():
    """Return a minimal hermes config dict suitable for unit tests."""
    return {
        "model": "test/mock-model",
        "toolsets": ["terminal", "file"],
        "max_turns": 10,
        "terminal": {
            "backend": "local",
            "cwd": "/tmp",
            "timeout": 30,
        },
        "compression": {"enabled": False},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "command_allowlist": [],
    }


# ── Global test timeout ─────────────────────────────────────────────────────
# Kill any individual test that takes longer than 30 seconds.
# Prevents hanging tests (subprocess spawns, blocking I/O) from stalling the
# entire test suite.

def _timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded 30 second timeout")

@pytest.fixture(autouse=True)
def _ensure_current_event_loop(request):
    """Provide a default event loop for sync tests that call get_event_loop().

    Python 3.11+ no longer guarantees a current loop for plain synchronous tests.
    A number of gateway tests still use asyncio.get_event_loop().run_until_complete(...).
    Ensure they always have a usable loop without interfering with pytest-asyncio's
    own loop management for @pytest.mark.asyncio tests.
    """
    if request.node.get_closest_marker("asyncio") is not None:
        yield
        return

    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        loop = None

    created = loop is None or loop.is_closed()
    if created:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        yield
    finally:
        if created and loop is not None:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)


@pytest.fixture(autouse=True)
def _enforce_test_timeout():
    """Kill any individual test that takes longer than 30 seconds.
    SIGALRM is Unix-only; skip on Windows."""
    if sys.platform == "win32":
        yield
        return
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(30)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)


@pytest.fixture(autouse=True)
def _isolate_environment():
    """Snapshot/restore os.environ per test to prevent cross-test pollution under xdist."""
    saved = os.environ.copy()
    try:
        yield
    finally:
        # Restore: remove added keys, re-add deleted keys, reset mutated values
        current_keys = set(os.environ.keys())
        saved_keys = set(saved.keys())
        for k in current_keys - saved_keys:
            os.environ.pop(k, None)
        for k in saved_keys - current_keys:
            os.environ[k] = saved[k]
        for k in saved_keys & current_keys:
            if os.environ[k] != saved[k]:
                os.environ[k] = saved[k]


@pytest.fixture(autouse=True)
def _isolate_models_dev_cache():
    """Snapshot/restore agent.models_dev module-level cache to prevent test pollution.

    Some tests overwrite ``_models_dev_cache`` with a synthetic registry that
    lacks providers like ``opencode-go``. Without isolation, downstream tests
    on the same xdist worker observe the polluted cache and fail intermittently.
    """
    try:
        from agent import models_dev as _md
    except Exception:
        yield
        return

    saved_cache = getattr(_md, "_models_dev_cache", None)
    saved_time = getattr(_md, "_models_dev_cache_time", None)
    try:
        yield
    finally:
        if hasattr(_md, "_models_dev_cache"):
            _md._models_dev_cache = saved_cache
        if hasattr(_md, "_models_dev_cache_time"):
            _md._models_dev_cache_time = saved_time


def pytest_configure(config):
    """Eagerly import tool modules so the global registry is populated regardless
    of which tests run on a given xdist worker. Without this, tests like
    test_terminal_tool_present fail when scheduled on a worker where no other
    test has imported tools.terminal_tool. Failures here are non-fatal because
    some environments lack optional native deps used by individual tool modules.
    """
    for mod in ("tools.terminal_tool", "tools.file_tools"):
        try:
            __import__(mod)
        except Exception:
            # Tool module may be unavailable in some environments; tests that
            # require it will skip or fail with clearer errors than import-time.
            pass
