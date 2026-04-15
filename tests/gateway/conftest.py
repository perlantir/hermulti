"""Gateway test fixtures.

Isolates module-level approval state across tests so pytest-xdist workers
don't observe torn state from sibling approval tests.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_approval_state(request):
    """Clear tools.approval module-level state before and after each test.

    Scoped narrowly to approval-related tests so unrelated gateway tests
    don't get their state wiped.
    """
    path = str(request.node.path) if hasattr(request.node, "path") else ""
    if "approve_deny" not in path and "approval" not in path.lower():
        yield
        return

    from tools import approval as mod

    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()
    try:
        yield
    finally:
        mod._gateway_queues.clear()
        mod._gateway_notify_cbs.clear()
        mod._session_approved.clear()
        mod._permanent_approved.clear()
        mod._pending.clear()
