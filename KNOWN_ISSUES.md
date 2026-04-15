# Known Issues

This document tracks tests that are intentionally skipped because they require
external resources or test removed behavior that is not currently planned.

## Skipped tests

### tests/cron/test_jobs.py (4 tests)
Skipped: require the optional `croniter` package, which is not installed in the
default dev environment. Install `croniter` to run cron-job scheduling tests:
`pip install croniter`.

