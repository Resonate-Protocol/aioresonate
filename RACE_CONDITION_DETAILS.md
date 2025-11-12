# Detailed Race Condition Analysis: aioresonate Client Disconnect/Reconnect

## Executive Summary

This analysis identifies **8 critical and high-severity race conditions** in the aioresonate codebase related to client disconnect/reconnect scenarios. These issues are most likely to manifest when clients:
1. Rapidly disconnect and reconnect
2. Experience network timeouts while operations are pending
3. Call methods concurrent with disconnect/reconnect
4. Have async callbacks executing during state transitions

## 1. CLIENT-SIDE RACE CONDITIONS

### RC-1: Non-atomic `connected` property check leading to stale state (HIGH SEVERITY)

**Location**: `/home/user/aioresonate/aioresonate/client/client.py`, lines 238-240, 685

**Code**:
```python
@property
def connected(self) -> bool:
    return self._connected and self._ws is not None and not self._ws.closed
```

**Race Condition**:
- The `connected` property performs 3 separate checks without atomicity
- Between checks, another task could modify `_connected`, `_ws`, or close the WebSocket
- `_time_sync_loop()` at line 685 uses this property in a loop: `while self.connected:`

**Scenario**:
1. Thread A: Checks `self._connected` → True
2. Thread B: Sets `self._connected = False` in `disconnect()`
3. Thread A: Checks `self._ws is not None` → Still true (not yet cleared)
4. Thread A: Enters loop iteration thinking connection is valid
5. Thread A: Loop iteration uses stale `self._ws` reference

**Impact**:
- `_time_sync_loop()` may continue after disconnect
- Operations using stale WebSocket reference
- AttributeError or ConnectionError in time sync code

**Code Locations**:
- Property definition: line 240
- Usage in time_sync_loop: line 685
- Usage in send_player_state: line 323
- Usage in send_group_command: line 338
- Usage in _send_time_message: line 400

---

### RC-2: Reader loop calls disconnect while disconnect is modifying _connected (HIGH SEVERITY)

**Location**: `/home/user/aioresonate/aioresonate/client/client.py`, lines 412-423 and 282-313

**Code**:
```python
# _reader_loop (lines 412-423)
async def _reader_loop(self) -> None:
    assert self._ws is not None
    try:
        async for msg in self._ws:
            await self._handle_ws_message(msg)
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("WebSocket reader encountered an error")
    finally:
        if self._connected:  # LINE 422 - Race here!
            await self.disconnect()

# disconnect() (lines 282-313)
async def disconnect(self) -> None:
    self._connected = False  # LINE 284 - Modifies state
    current_task = asyncio.current_task(loop=self._loop)

    if self._time_task is not None and self._time_task is not current_task:
        self._time_task.cancel()
        # ... more operations
```

**Race Condition**:
- Reader loop checks `self._connected` at line 422 (not atomically protected)
- Disconnect can be called from other tasks and sets `_connected = False`
- Multiple concurrent calls to `disconnect()` can happen

**Scenario**:
1. Normal disconnect initiated externally
2. Sets `_connected = False`
3. Reader loop finishes at nearly same time
4. Reader loop checks `self._connected` and it's already False
5. Reader loop doesn't call `disconnect()` - cleanup might be skipped
6. OR: Both call disconnect simultaneously

**Impact**:
- Cleanup skipped or duplicated
- Tasks not properly cancelled
- Time filter not reset
- Callbacks not invoked
- Resources not released properly

**Code Locations**:
- Reader loop check: line 422
- Disconnect start: line 284

---

### RC-3: Time sync loop exit race with concurrent callback accesses (MEDIUM SEVERITY)

**Location**: `/home/user/aioresonate/aioresonate/client/client.py`, lines 633-681

**Issue**:
Callbacks can be set/cleared (lines 346-378) while being invoked in notification methods (lines 633-681). The callbacks are not protected by locks.

**Scenario**:
1. Reader task is processing message and calls `_notify_metadata_callback()`
2. Main task calls `set_metadata_listener(None)` to clear callback
3. Between the None check and callback invocation, callback is cleared
4. Code still tries to invoke cleared callback (None)

**Code locations**:
- Callbacks stored: lines 160-171
- Callbacks cleared: lines 346, 350, 354, 358, 374, 378
- Callbacks invoked: lines 634-681

---

### RC-4: Concurrent send_message and disconnect on _send_lock (MEDIUM SEVERITY)

**Location**: `/home/user/aioresonate/aioresonate/client/client.py`, lines 406-410 and 282-313

**Code**:
```python
async def _send_message(self, payload: str) -> None:
    if not self._ws:  # Non-atomic check!
        raise RuntimeError("WebSocket is not connected")
    async with self._send_lock:
        await self._ws.send_str(payload)

async def disconnect(self) -> None:
    self._connected = False
    # ... later
    if self._ws is not None:
        await self._ws.close()
        self._ws = None
```

**Race Condition**:
1. Task A: `_send_message()` checks `if not self._ws:` → passes
2. Task B: `disconnect()` is called, sets `self._ws = None`
3. Task A: Acquires `_send_lock`
4. Task A: Tries to send on now-None `_ws`

**Impact**:
- AttributeError trying to send on None
- Sent data lost
- Connection appears to work but isn't

**Code Locations**:
- Check: line 407
- Send: line 410
- Disconnect clearing: line 300

---

### RC-5: _send_message called after disconnect with non-None ws (LOW SEVERITY)

**Location**: `/home/user/aioresonate/aioresonate/client/client.py`, lines 399-410

**Code**:
```python
async def _send_time_message(self) -> None:
    if not self.connected:  # Non-atomic check!
        return
    now_us = self._now_us()
    message = ClientTimeMessage(...)
    await self._send_message(message.to_json())
```

**Scenario**:
1. `_time_sync_loop` at line 685: `while self.connected:`
2. Loop iteration checks `self.connected` → True
3. Calls `_send_time_message()` which re-checks at line 400 → True
4. Meanwhile disconnect is initiated
5. By the time `_send_message()` executes, `_ws` could be closed or None

**Impact**: Same as RC-4

---

### RC-6: Multiple tasks writing to _time_filter without synchronization (MEDIUM SEVERITY)

**Location**: `/home/user/aioresonate/aioresonate/client/time_sync.py` and `/home/user/aioresonate/aioresonate/client/client.py`

**Issue**:
- `_time_filter.update()` is called from `_handle_server_time()` (line 505)
- `compute_client_time()` and `compute_server_time()` are called from callbacks in `_handle_audio_chunk()` (line 589)
- These can happen concurrently

**Time Filter Methods Involved**:
- `update()` at line 63: Modifies `_offset`, `_drift`, `_current_time_element`
- `compute_client_time()` at line 224: Reads `_current_time_element`
- `compute_server_time()` at line 197: Reads `_current_time_element`

**Scenario**:
1. Reader task processes ServerTimeMessage
2. Calls `_handle_server_time()` which calls `_time_filter.update()`
3. Meanwhile, audio chunk callback is executing
4. Callback calls `compute_play_time()` which calls `_time_filter.compute_client_time()`
5. Time filter is in middle of update - reads inconsistent state

**Impact**:
- Incorrect audio timing
- Audio chunks played at wrong times
- Jumps in playback time

---

## 2. SERVER-SIDE CLIENT RACE CONDITIONS

### RC-7: Multiple concurrent disconnect calls on same client (HIGH SEVERITY)

**Location**: `/home/user/aioresonate/aioresonate/server/client.py`, lines 168-200

**Code**:
```python
async def disconnect(self, *, retry_connection: bool = True) -> None:
    """Disconnect this client from the server."""
    if not retry_connection:
        self._closing = True
    self._disconnecting = True  # Not atomic!

    # ... cleanup operations

    if self._client_id is not None:
        self._handle_client_disconnect(self)  # Can be called multiple times!
```

**Race Condition**:
- `_disconnecting` flag is checked at line 172 but not atomically
- Multiple code paths can call `disconnect()`:
  1. `_cleanup_connection()` at line 415
  2. `send_message()` at line 557 (when queue full)
  3. External callers

**Scenario**:
1. Writer task fails, calls `disconnect()`
2. Message loop fails simultaneously, also calls `disconnect()`
3. Both threads execute past the `_disconnecting` check
4. Both attempt to cancel writer task
5. Both call `_handle_client_disconnect(self)` twice

**Impact**:
- Callbacks invoked multiple times
- Client added/removed from group multiple times
- Resource cleanup duplicated
- Unpredictable server state

**Code Locations**:
- Flag check: line 172
- Disconnect callback: line 198
- Called from: lines 415, 557

---

### RC-8: Writer task completion between message loop check and usage (MEDIUM SEVERITY)

**Location**: `/home/user/aioresonate/aioresonate/server/client.py`, lines 352-375

**Code**:
```python
async def _run_message_loop(self) -> None:
    while not wsock.closed:
        # Create receive task
        receive_task = self._server.loop.create_task(wsock.receive())
        assert self._writer_task is not None  # LINE 363
        done, pending = await asyncio.wait(
            [receive_task, self._writer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if self._writer_task in done:
            # Writer ended, need to cleanup
            break
```

**Race Condition**:
- Writer task can complete between line 363 check and line 364-366 wait
- Though wait() handles this, the assert at line 363 could fail if writer task has been set to None from another disconnect path

**Impact**: Lower priority than others but could cause assertion failures

---

## 3. SERVER CONNECTION MANAGEMENT RACE CONDITIONS

### RC-9: Non-atomic connect_to_client state check-and-create (HIGH SEVERITY)

**Location**: `/home/user/aioresonate/aioresonate/server/server.py`, lines 150-170

**Code**:
```python
def connect_to_client(self, url: str) -> None:
    logger.debug("Connecting to client at URL: %s", url)
    prev_task = self._connection_tasks.get(url)  # Non-atomic!
    if prev_task is not None:
        logger.debug("Connection is already active for URL: %s", url)
        if retry_event := self._retry_events.get(url):
            retry_event.set()
    else:
        # Create retry event for this connection
        self._retry_events[url] = asyncio.Event()
        self._connection_tasks[url] = self._loop.create_task(
            self._handle_client_connection(url)
        )
```

**Race Condition**:
- Check at line 158 and creation at lines 167-169 are not atomic
- Dictionary is shared with `_handle_client_connection()` which modifies it at line 252

**Scenario**:
1. Task A: `connect_to_client(url)` - checks line 158, gets None
2. Task B: `connect_to_client(url)` - same check, also gets None
3. Task A: Creates task and stores at line 168
4. Task B: Creates task and overwrites at line 168
5. Task A's connection task is lost, orphaned

**Impact**:
- Connection attempt silently lost
- First connection task never finishes, never cleanup
- Task leak
- Multiple simultaneous connections created

**Code Locations**:
- Check: line 158
- Create: lines 167-169
- Cleanup: line 252

---

### RC-10: Stale retry_event reference due to race with _handle_client_connection cleanup (MEDIUM SEVERITY)

**Location**: `/home/user/aioresonate/aioresonate/server/server.py`, lines 195, 231-237, 252-253

**Code**:
```python
# In _handle_client_connection
while True:
    retry_event = self._retry_events.get(url)  # LINE 195 - Gets reference

    # ... connection attempt ...

    if retry_event is not None:  # LINE 231
        try:
            await asyncio.wait_for(retry_event.wait(), timeout=backoff)
            # ...
            retry_event.clear()  # LINE 237
        except TimeoutError:
            pass
    else:
        await asyncio.sleep(backoff)

finally:
    self._retry_events.pop(url, None)  # LINE 253 - Removes from dict
```

**Race Condition**:
1. Connection task gets `retry_event` reference at line 195
2. Meanwhile, `disconnect_from_client()` cancels the task
3. Task cleanup at line 253 pops the event from dict
4. But connect_to_client() could be called again, creating NEW event
5. First task still has reference to OLD event
6. First task calls `retry_event.clear()` on old event
7. New connection task is not properly signaled

**Scenario**:
1. Connection established to URL
2. Call `disconnect_from_client(url)` - cancels task
3. Task cleanup removes retry_event
4. Immediately call `connect_to_client(url)` again
5. Creates new retry_event in dict
6. Cancelled task's exception cleanup still references old event
7. New task's retry_event never gets set/cleared properly

**Impact**:
- Retry signaling broken
- New connection doesn't respond to immediate retry requests
- Connection waits full backoff period unnecessarily

---

## Summary Table of Race Conditions

| ID | Component | Severity | Issue | Fix Complexity |
|----|-----------|----------|-------|-----------------|
| RC-1 | Client | HIGH | Non-atomic connected property | Medium |
| RC-2 | Client | HIGH | Multiple disconnect calls | Medium |
| RC-3 | Client | MEDIUM | Callback access without locks | Low |
| RC-4 | Client | MEDIUM | Send after disconnect race | Medium |
| RC-5 | Client | LOW | Stale ws reference | Low |
| RC-6 | Client | MEDIUM | Time filter concurrent access | Medium |
| RC-7 | Server Client | HIGH | Multiple disconnect calls | Medium |
| RC-8 | Server Client | MEDIUM | Task reference race | Low |
| RC-9 | Server Connection | HIGH | Non-atomic check-and-create | Medium |
| RC-10 | Server Connection | MEDIUM | Stale event reference | Medium |

## Critical Recommendations

1. **Implement locks** for shared state access in both client and server
2. **Make state transitions atomic** using proper synchronization primitives
3. **Add double-check locking** for initialization patterns
4. **Use compare-and-swap semantics** for task/callback management
5. **Add unit tests** for concurrent disconnect/reconnect scenarios
6. **Review callback invocation** - ensure no stale references
