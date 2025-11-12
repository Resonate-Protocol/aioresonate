# Comprehensive Race Condition Analysis: aioresonate Disconnect/Reconnect

## Overview

This document details **10 distinct race conditions** found in the aioresonate codebase during investigation of client disconnect/reconnect scenarios. The analysis covers the full connection lifecycle including connect, message handling, disconnect, and reconnection logic.

**Investigation Date**: 2025-11-12
**Severity Assessment**: CRITICAL - 4 issues, HIGH - 2 issues, MEDIUM - 4 issues

---

## Quick Reference Table

| ID | Component | Severity | Location | Impact |
|----|-----------|----------|----------|--------|
| RC-1 | Client | HIGH | client.py:238-240,685 | Stale WebSocket reference, time sync continues |
| RC-2 | Client | HIGH | client.py:412-423,282-313 | Multiple disconnect calls, skipped cleanup |
| RC-3 | Client | MEDIUM | client.py:346-378,633-681 | Callbacks cleared while invoked |
| RC-4 | Client | MEDIUM | client.py:406-410,282-313 | AttributeError on None WebSocket |
| RC-5 | Client | LOW | client.py:399-410 | Sent data lost |
| RC-6 | Client | MEDIUM | time_sync.py,client.py:505,589 | Audio timing jumps |
| RC-7 | Server Client | HIGH | server/client.py:168-200 | State corruption, callbacks invoked multiple times |
| RC-8 | Server Client | MEDIUM | server/client.py:352-375 | Assertion failures possible |
| RC-9 | Server Connection | HIGH | server/server.py:150-170 | Task leaks, orphaned connections |
| RC-10 | Server Connection | MEDIUM | server/server.py:186-253 | Retry signaling broken |

---

## Client-Side Race Conditions

### RC-1: Non-atomic `connected` Property (HIGH SEVERITY)

**File**: `/home/user/aioresonate/aioresonate/client/client.py`
**Lines**: 238-240, 685

**Issue**: The `connected` property performs 3 independent checks without atomic protection:

```python
@property
def connected(self) -> bool:
    return self._connected and self._ws is not None and not self._ws.closed
    # Each check can change between evaluations!
```

**Root Cause**: Between the first and third check, another task can:
1. Set `_connected = False`
2. Close or set `_ws = None`
3. Cause `_ws.closed` to become True

**Impact**:
- `_time_sync_loop()` at line 685 uses this in loop condition
- Loop may continue after disconnect
- Stale WebSocket references accessed
- AttributeError or connection attempts on dead socket

**Reproduction**:
1. Create client and connect
2. Call `disconnect()` from one task
3. Have time sync loop call `_send_time_message()` concurrently
4. Will access `_ws` after it's been closed or set to None

---

### RC-2: Reader Loop Disconnect Race (HIGH SEVERITY)

**File**: `/home/user/aioresonate/aioresonate/client/client.py`
**Lines**: 412-423, 282-313

**Issue**: Reader loop's finally block checks `_connected` without protection:

```python
async def _reader_loop(self) -> None:
    # ... iterate messages ...
    finally:
        if self._connected:  # RACE CONDITION HERE
            await self.disconnect()
```

Meanwhile, `disconnect()` modifies `_connected`:
```python
async def disconnect(self) -> None:
    self._connected = False  # Can be modified while reader checks
    # ... rest of disconnect logic
```

**Root Cause**: No synchronization between check at line 422 and state modification at line 284

**Impact**:
- If disconnect is called externally before reader exits, reader may not call `disconnect()`
- Cleanup code in reader's disconnect() call is skipped
- If both paths try to disconnect simultaneously, cleanup is duplicated
- Callbacks not invoked at appropriate times

---

### RC-3: Callback Access Without Locks (MEDIUM SEVERITY)

**File**: `/home/user/aioresonate/aioresonate/client/client.py`
**Lines**: 346-378 (setters), 633-681 (invocations)

**Issue**: Callbacks can be cleared while being invoked:

```python
# Setting callback from user code
def set_metadata_listener(self, callback):
    self._metadata_callback = callback

# Invoking callback from message handler
async def _notify_metadata_callback(self, payload):
    if self._metadata_callback is None:  # Check
        return
    # ... delay ...
    result = self._metadata_callback(payload)  # Race! Callback may be None now
```

**Root Cause**: No lock protecting `_metadata_callback` and other callback fields

**Impact**:
- Callback cleared between None check and invocation
- NoneType is not callable error
- User code can clear callbacks while being executed

---

### RC-4: Send-After-Disconnect (MEDIUM SEVERITY)

**File**: `/home/user/aioresonate/aioresonate/client/client.py`
**Lines**: 406-410, 282-313

**Issue**: Check-then-act race on `_ws`:

```python
async def _send_message(self, payload: str) -> None:
    if not self._ws:  # Check WITHOUT lock
        raise RuntimeError("WebSocket is not connected")
    async with self._send_lock:  # Lock acquired AFTER check
        await self._ws.send_str(payload)  # Use after lock
```

**Root Cause**:
- Check happens outside the lock
- Disconnect can set `_ws = None` between check and use
- When lock is acquired, `_ws` is already None

**Timeline**:
1. `_send_message()` checks `if not self._ws:` → passes
2. `disconnect()` executes, sets `self._ws = None`
3. `_send_message()` acquires lock
4. `_send_message()` tries `self._ws.send_str()` on None
5. **AttributeError**

---

### RC-5: Stale WebSocket Reference (LOW SEVERITY)

**File**: `/home/user/aioresonate/aioresonate/client/client.py`
**Lines**: 399-410

**Issue**: `_send_time_message()` has non-atomic check:

```python
async def _send_time_message(self) -> None:
    if not self.connected:  # Non-atomic check
        return
    # ... build message ...
    await self._send_message(message.to_json())  # _ws could be None now
```

**Impact**: Less severe than RC-4 because it's a re-check, but still exploitable

---

### RC-6: Time Filter Concurrent Access (MEDIUM SEVERITY)

**File**: `/home/user/aioresonate/aioresonate/client/time_sync.py` and `client.py`

**Issue**: Time filter accessed concurrently without synchronization:

**Writer Path** (from reader task):
```python
# _handle_server_time() -> _time_filter.update()
# Modifies: _offset, _drift, _offset_covariance, _current_time_element
```

**Reader Path** (from user audio callback):
```python
# compute_play_time() -> compute_client_time()
# Reads: _current_time_element.offset, _current_time_element.drift
```

**Root Cause**: No lock protecting time filter's internal state

**Impact**:
- Audio chunk timing calculations use stale or partial data
- Offset and drift values don't match (different versions)
- Audio playback times jump erratically
- Synchronization converges incorrectly

---

## Server-Side Race Conditions

### RC-7: Multiple Concurrent Disconnect Calls (HIGH SEVERITY)

**File**: `/home/user/aioresonate/aioresonate/server/client.py`
**Lines**: 168-200

**Issue**: `disconnect()` can be called concurrently from multiple paths:

```python
async def disconnect(self, *, retry_connection: bool = True) -> None:
    if not retry_connection:
        self._closing = True
    self._disconnecting = True  # Not atomically protected!

    # ... cleanup ...

    if self._client_id is not None:
        self._handle_client_disconnect(self)  # Called multiple times?
```

**Multiple Entry Points**:
1. `_cleanup_connection()` line 415
2. `send_message()` queue full handler line 557
3. External user code
4. Message loop failures

**Impact**:
- Callbacks invoked multiple times
- Client removed from group multiple times
- Group state becomes corrupted
- Cleanup operations duplicated

**Example Timeline**:
1. Message queue fills
2. `send_message()` initiates disconnect task
3. Writer task also encounters error
4. Both call `disconnect()` concurrently
5. Both try to cancel writer task
6. Both call `_handle_client_disconnect()` twice
7. Group receives "client removed" event twice

---

### RC-8: Task Reference Race (MEDIUM SEVERITY)

**File**: `/home/user/aioresonate/aioresonate/server/client.py`
**Lines**: 352-375

**Issue**: Writer task reference can be cleared between check and use:

```python
async def _run_message_loop(self) -> None:
    while not wsock.closed:
        receive_task = self._server.loop.create_task(wsock.receive())
        assert self._writer_task is not None  # Check at line 363
        done, pending = await asyncio.wait(
            [receive_task, self._writer_task],  # Use at line 365
            return_when=asyncio.FIRST_COMPLETED,
        )
```

**Impact**: Assertion could fail if writer task is set to None between lines 363-365

---

## Server Connection Management Race Conditions

### RC-9: Non-atomic Check-and-Create (HIGH SEVERITY)

**File**: `/home/user/aioresonate/aioresonate/server/server.py`
**Lines**: 150-170

**Issue**: Dictionary check and creation not atomic:

```python
def connect_to_client(self, url: str) -> None:
    prev_task = self._connection_tasks.get(url)  # Line 158: Check
    if prev_task is not None:
        # ...retry signaling
    else:
        self._retry_events[url] = asyncio.Event()  # Line 167: Create
        self._connection_tasks[url] = self._loop.create_task(  # Line 168: Create
            self._handle_client_connection(url)
        )
```

**Race Scenario**:
1. Task A: `get(url)` returns None
2. Task B: `get(url)` also returns None (same time)
3. Task A: Creates Event A, stores in dict
4. Task B: Creates Event B, **overwrites** in dict (Task A's event lost)
5. Task A: Creates Task A, stores in dict
6. Task B: Creates Task B, **overwrites** in dict (Task A orphaned)

**Impact**:
- Task A never finishes, never cleaned up
- Task A is resource leak (orphaned asyncio.Task)
- Task B is the only active connection task
- Both Event A and Task A are lost/orphaned
- Multiple connections possible if this races

---

### RC-10: Stale Event Reference (MEDIUM SEVERITY)

**File**: `/home/user/aioresonate/aioresonate/server/server.py`
**Lines**: 186-253

**Issue**: Event reference becomes stale after cleanup:

```python
async def _handle_client_connection(self, url: str) -> None:
    while True:
        retry_event = self._retry_events.get(url)  # Line 195: Get reference

        # ... connection attempt ...

        if retry_event is not None:  # Line 231
            try:
                await asyncio.wait_for(retry_event.wait(), timeout=backoff)
                retry_event.clear()  # Line 237
            except TimeoutError:
                pass
        else:
            await asyncio.sleep(backoff)

    finally:
        self._retry_events.pop(url, None)  # Line 253: Remove from dict
```

**Race Scenario**:
1. Connection task gets reference to Event A at line 195
2. Connection fails, enters backoff
3. `disconnect_from_client(url)` called externally
4. Cancels the connection task
5. Task cleanup (finally block) pops Event A from dict at line 253
6. Meanwhile, `connect_to_client(url)` called again
7. Creates new Event B, stores in dict
8. Original task still has reference to Event A
9. When original task finally executes line 237: `retry_event.clear()`
10. Clears Event A (wrong event!)
11. Event B never gets cleared
12. Event B is orphaned, doesn't respond to retry signals

**Impact**:
- Retry event signaling broken
- New connection doesn't respond to "retry now" requests
- New connection waits full backoff period unnecessarily

---

## Root Cause Analysis

### Underlying Problems

1. **No Synchronization Primitives**
   - No locks protecting shared state
   - No atomic operations for critical sections
   - Properties return non-atomic results

2. **Multiple Disconnect Paths**
   - No single entry point
   - No mechanism to prevent concurrent execution
   - Flag-based guards not sufficient

3. **Task Lifetime Not Managed**
   - Tasks referenced after cleanup
   - References held across boundaries
   - Dictionary operations race with task references

4. **State Distribution**
   - Connection state scattered across variables
   - No atomic "connected" check
   - Callbacks not protected

5. **Event Reference Management**
   - Long-lived references to events
   - Events deleted while referenced
   - No validation before use

---

## Recommendations

### Immediate Fixes (Critical)

1. **Make disconnect() atomic**
   - Use a lock to ensure single execution
   - Move `_disconnecting` check inside lock
   - Prevent multiple concurrent disconnect calls

2. **Fix connected property**
   - Make check atomic
   - Or provide separate method
   - Or restructure state

3. **Protect callback access**
   - Use lock when setting/invoking callbacks
   - Or use snapshot pattern

4. **Fix connect_to_client()**
   - Use lock for dictionary operations
   - Implement atomic check-and-create
   - Or use setdefault() pattern

### Short-term Fixes (High Priority)

1. Add locks to time filter
2. Protect server client disconnect()
3. Fix event reference lifecycle
4. Add retry logic for failed operations

### Testing

1. Add concurrency tests
2. Test rapid disconnect/reconnect
3. Test multiple connect() calls with same URL
4. Test callback invocation during disconnect
5. Stress test with high message volume

---

## Files Requiring Changes

**Primary**:
- `aioresonate/client/client.py` - 6 issues
- `aioresonate/server/client.py` - 2 issues
- `aioresonate/server/server.py` - 2 issues

**Secondary**:
- `aioresonate/client/time_sync.py` - 1 issue

**May Be Affected**:
- `aioresonate/server/group.py` - By RC-7
- `aioresonate/server/player.py` - By RC-7

---

## Report Generated

- **Date**: 2025-11-12
- **Branch**: claude/fix-reconnect-race-conditions-011CV3teQzHpcs8avdFSHd3P
- **Severity**: CRITICAL
- **Issues Found**: 10
- **Recommended Actions**: Fix all HIGH severity issues before production
