# Route 2 Runtime Execution Check

Purpose: verify that workflow checks use structured runtime execution evidence.

Pass conditions:
- runtime results include command, cwd, env overrides, stdout, stderr, exit code, timeout, and duration
- execution uses UTF-8-safe output capture
- workspace-root cwd discipline is preserved unless explicitly overridden

Primary test coverage:
- `tests/test_route2_mature_loop.py::test_runtime_exec_reports_success_failure_timeout_unicode_and_env`
