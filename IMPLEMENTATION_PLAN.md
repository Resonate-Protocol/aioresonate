# Plan: Fix Restart/Disconnect Races in aioresonate

This plan reflects a fresh code review of the repository to verify and refine the analysis items. The biggest correctness wins come from making disconnect paths idempotent and moving send checks under the send lock. Some items from the analysis are unnecessary under asyncio’s single-threaded semantics and are adjusted accordingly.

## Phase 1: Critical Fixes (High Priority)

1) Client disconnect idempotency and lifecycle
File: `aioresonate/client/client.py`
- Add `_lifecycle_lock: asyncio.Lock` and `_disconnecting/_disconnected: bool` flags.
- Make `disconnect()` idempotent: the first caller performs cleanup; concurrent callers await the same path and return.
- Reader loop `finally`: call `await self.disconnect()` unconditionally (idempotency prevents duplicate cleanup) or guard with `_disconnecting`.
- Leave `connected` property simple; rely on idempotent disconnect + robust send checks to avoid stale use.

2) Send-after-disconnect robustness
File: `aioresonate/client/client.py`
- Move the WebSocket presence/closed checks inside `_send_lock`.
- Capture `ws = self._ws` under the lock and use the local variable for the send.
- When not connected, raise a clear error without touching `self._ws` after the check.

3) Server client disconnect idempotency
File: `aioresonate/server/client.py`
- Add `_disconnect_lock: asyncio.Lock` and `_disconnected: bool`.
- Make `disconnect()` idempotent: cancel writer once, close ws once, invoke disconnect callback once.
- Keep writer-task cancellation and queue draining safe against concurrent callers.

4) Guard server connect/disconnect task maps with a lock
File: `aioresonate/server/server.py`
- Add `_conn_lock: asyncio.Lock`.
- Wrap both `connect_to_client()` and `disconnect_from_client()` updates to `_connection_tasks` and `_retry_events` with `_conn_lock`.
- Continue to set the retry event when a task exists to achieve immediate retry; prevent duplicate task creation.

## Phase 2: Medium Priority Fixes

5) Callback invocation safety via snapshot (no locks)
File: `aioresonate/client/client.py`
- Snapshot the callback into a local variable, check for `None`, then invoke; this avoids surprises if user code clears the callback while you’re logging errors.
- Full locking is unnecessary here because these accessors and invocations don’t `await` between check and call.

6) Time filter access consistency via snapshot (no locks)
File: `aioresonate/client/time_sync.py`
- Update `compute_client_time()` and `compute_server_time()` to snapshot `elem = self._current_time_element` once and read from it; `update()`/compute methods are synchronous (no `await`) so locks are not needed under asyncio.

7) Server writer-task guard (defensive)
File: `aioresonate/server/client.py`
- In `_run_message_loop()`, replace the strict `assert self._writer_task is not None` with a defensive check: if missing or done, break and clean up.

Notes on RCs adjusted after code review
- RC-3 (callback access) and RC-6 (time filter) do not require locks under asyncio since methods don’t `await`; snapshots suffice.
- RC-9/RC-10 (server connect/task/event maps): while asyncio scheduling makes “check-then-create” less problematic, using `_conn_lock` across connect/disconnect keeps the model simple and prevents edge-case churn when task cleanup and new connections overlap.

## Phase 3: Testing & Validation

8) Targeted concurrency tests
- Rapid connect/disconnect on the client; ensure reader `finally` + external `disconnect()` do not double-cleanup and callbacks fire exactly once.
- Send during disconnect: verify `_send_message` never uses a cleared socket and raises predictable errors.
- Server queue-full → disconnect → reconnect path: only one disconnect occurs and server state remains consistent.
- Double `connect_to_client(url)` calls: only one task exists; retry signaling wakes backoff immediately.

9) Pre-commit and type checks
- Run `./scripts/run-in-env.sh pre-commit run -a` and fix lint/type issues.

10) Documentation cleanup
- Keep the analysis documents until fixes are merged/reviewed; then prune or update them to reflect the final state.

## Summary

Files to modify
- `aioresonate/client/client.py`
- `aioresonate/server/client.py`
- `aioresonate/server/server.py`
- `aioresonate/client/time_sync.py` (snapshot reads only)

Estimated effort: 6–10 hours
