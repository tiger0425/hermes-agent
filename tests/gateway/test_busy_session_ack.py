"""Tests for busy-session acknowledgment when user sends messages during active agent runs.

Verifies that users get an immediate status response instead of total silence
when the agent is working on a task. See PR fix for the @Lonely__MH report.
"""
import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so we can import gateway code without heavy deps
# ---------------------------------------------------------------------------
import sys, types

_tg = types.ModuleType("telegram")
_tg_constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
setattr(_tg, "constants", _tg_constants)
setattr(_tg_constants, "ChatType", _ct)
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg_constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)
from gateway.config import Platform
from gateway.run import (
    _load_week9_formal_release_binding_metadata,
    _resolve_week9_release_freeze_record_path,
    get_current_clarify_binding_metadata,
    reset_current_clarify_binding_metadata,
    set_current_clarify_binding_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(text="hello", chat_id="123", platform_val="telegram"):
    """Build a minimal MessageEvent."""
    source = SessionSource(
        platform=MagicMock(value=platform_val),
        chat_id=chat_id,
        chat_type="private",
        user_id="user1",
    )
    evt = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
    )
    return evt


def _make_runner():
    """Build a minimal GatewayRunner-like object for testing."""
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_clarify = {}
    runner._clarify_resolution_state = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    return runner, _AGENT_PENDING_SENTINEL


def _make_adapter(platform_val="telegram"):
    """Build a minimal adapter mock."""
    class _AdapterMock(MagicMock):
        async def edit_message(self, *args, **kwargs):
            return None

    adapter = _AdapterMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value=platform_val)
    return adapter


def _attach_immediate_formal_release_clarify(adapter, runner, session_key, decision="✅ Approve — proceed with Week 9 release"):
    async def _send_clarify_prompt(*args, **kwargs):
        runner._resolve_pending_clarify(
            session_key,
            decision,
            outcome="answered",
        )
        return SimpleNamespace(success=True, message_id=f"clarify-{kwargs['clarify_id']}")

    setattr(type(adapter), "send_clarify_prompt", _send_clarify_prompt)
    return adapter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBusySessionAck:
    """User sends a message while agent is running — should get acknowledgment."""

    @pytest.mark.asyncio
    async def test_sends_ack_when_agent_running(self):
        """First message during busy session should get a status ack."""
        runner, sentinel = _make_runner()
        adapter = _make_adapter()

        event = _make_event(text="Are you working?")
        sk = build_session_key(event.source)

        # Simulate running agent
        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 21,
            "max_iterations": 60,
            "current_tool": "terminal",
            "last_activity_ts": time.time(),
            "last_activity_desc": "terminal",
            "seconds_since_activity": 1.0,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 600  # 10 min ago
        runner.adapters[event.source.platform] = adapter

        result = await runner._handle_active_session_busy_message(event, sk)

        assert result is True  # handled
        # Verify ack was sent
        adapter._send_with_retry.assert_called_once()
        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        if not content and call_kwargs.args:
            # positional args
            content = str(call_kwargs)
        assert "Interrupting" in content or "respond" in content
        assert "/stop" not in content  # no need — we ARE interrupting

        # Verify message was queued in adapter pending
        assert sk in adapter._pending_messages

        # Verify agent interrupt was called
        agent.interrupt.assert_called_once_with("Are you working?")

    @pytest.mark.asyncio
    async def test_debounce_suppresses_rapid_acks(self):
        """Second message within 30s should NOT send another ack."""
        runner, sentinel = _make_runner()
        adapter = _make_adapter()

        event1 = _make_event(text="hello?")
        # Reuse the same source so platform mock matches
        event2 = MessageEvent(
            text="still there?",
            message_type=MessageType.TEXT,
            source=event1.source,
            message_id="msg2",
        )
        sk = build_session_key(event1.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 5,
            "max_iterations": 60,
            "current_tool": None,
            "last_activity_ts": time.time(),
            "last_activity_desc": "api_call",
            "seconds_since_activity": 0.5,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 60
        runner.adapters[event1.source.platform] = adapter

        # First message — should get ack
        result1 = await runner._handle_active_session_busy_message(event1, sk)
        assert result1 is True
        assert adapter._send_with_retry.call_count == 1

        # Second message within cooldown — should be queued but no ack
        result2 = await runner._handle_active_session_busy_message(event2, sk)
        assert result2 is True
        assert adapter._send_with_retry.call_count == 1  # still 1, no new ack

        # But interrupt should still be called for both
        assert agent.interrupt.call_count == 2

    @pytest.mark.asyncio
    async def test_ack_after_cooldown_expires(self):
        """After 30s cooldown, a new message should send a fresh ack."""
        runner, sentinel = _make_runner()
        adapter = _make_adapter()

        event = _make_event(text="hello?")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 10,
            "max_iterations": 60,
            "current_tool": "web_search",
            "last_activity_ts": time.time(),
            "last_activity_desc": "tool",
            "seconds_since_activity": 0.5,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 120
        runner.adapters[event.source.platform] = adapter

        # First ack
        await runner._handle_active_session_busy_message(event, sk)
        assert adapter._send_with_retry.call_count == 1

        # Fake that cooldown expired
        runner._busy_ack_ts[sk] = time.time() - 31

        # Second ack should go through
        await runner._handle_active_session_busy_message(event, sk)
        assert adapter._send_with_retry.call_count == 2

    @pytest.mark.asyncio
    async def test_includes_status_detail(self):
        """Ack message should include iteration and tool info when available."""
        runner, sentinel = _make_runner()
        adapter = _make_adapter()

        event = _make_event(text="yo")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 21,
            "max_iterations": 60,
            "current_tool": "terminal",
            "last_activity_ts": time.time(),
            "last_activity_desc": "terminal",
            "seconds_since_activity": 0.5,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 600  # 10 min
        runner.adapters[event.source.platform] = adapter

        await runner._handle_active_session_busy_message(event, sk)

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content", "")
        assert "21/60" in content  # iteration
        assert "terminal" in content  # current tool
        assert "10 min" in content  # elapsed

    @pytest.mark.asyncio
    async def test_draining_still_works(self):
        """Draining case should still produce the drain-specific message."""
        runner, sentinel = _make_runner()
        runner._draining = True
        adapter = _make_adapter()

        event = _make_event(text="hello")
        sk = build_session_key(event.source)
        runner.adapters[event.source.platform] = adapter

        # Mock the drain-specific methods
        runner._queue_during_drain_enabled = lambda: False
        runner._status_action_gerund = lambda: "restarting"

        result = await runner._handle_active_session_busy_message(event, sk)
        assert result is True

        call_kwargs = adapter._send_with_retry.call_args
        content = call_kwargs.kwargs.get("content", "")
        assert "restarting" in content

    @pytest.mark.asyncio
    async def test_pending_sentinel_no_interrupt(self):
        """When agent is PENDING_SENTINEL, don't call interrupt (it has no method)."""
        runner, sentinel = _make_runner()
        adapter = _make_adapter()

        event = _make_event(text="hey")
        sk = build_session_key(event.source)

        runner._running_agents[sk] = sentinel
        runner._running_agents_ts[sk] = time.time()
        runner.adapters[event.source.platform] = adapter

        result = await runner._handle_active_session_busy_message(event, sk)
        assert result is True
        # Should still send ack
        adapter._send_with_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_adapter_falls_through(self):
        """If adapter is missing, return False so default path handles it."""
        runner, sentinel = _make_runner()

        event = _make_event(text="hello")
        sk = build_session_key(event.source)

        # No adapter registered
        runner._running_agents[sk] = MagicMock()

        result = await runner._handle_active_session_busy_message(event, sk)
        assert result is False  # not handled, let default path try


class TestPendingClarify:
    """Pending clarify prompts should consume replies and cancel cleanly."""

    def test_consumes_card_button_reply(self):
        runner, _sentinel = _make_runner()
        event = _make_event(text='/card button {"clarify_id":"clarify_1","clarify_choice":"同意执行"}')
        sk = build_session_key(event.source)

        response_queue = MagicMock()
        runner._pending_clarify[sk] = {
            "clarify_id": "clarify_1",
            "choices": ["同意执行", "需要修改"],
            "response_queue": response_queue,
        }

        resolved = runner._consume_pending_clarify(event, sk)

        assert resolved is True
        response_queue.put_nowait.assert_called_once_with("同意执行")
        assert sk not in runner._pending_clarify

    def test_rejects_card_choice_not_in_pending_options(self):
        runner, _sentinel = _make_runner()
        event = _make_event(text='/card button {"clarify_id":"clarify_1","clarify_choice":"伪造选项"}')
        sk = build_session_key(event.source)

        response_queue = MagicMock()
        runner._pending_clarify[sk] = {
            "clarify_id": "clarify_1",
            "choices": ["同意执行", "需要修改"],
            "response_queue": response_queue,
        }

        resolved = runner._consume_pending_clarify(event, sk)

        assert resolved is False
        response_queue.put_nowait.assert_not_called()
        assert sk in runner._pending_clarify

    def test_cancel_pending_clarify_pushes_cancel_response(self):
        runner, _sentinel = _make_runner()
        event = _make_event(text="hello")
        sk = build_session_key(event.source)

        response_queue = MagicMock()
        runner._pending_clarify[sk] = {
            "clarify_id": "clarify_2",
            "choices": ["A", "B"],
            "response_queue": response_queue,
        }

        cancelled = runner._cancel_pending_clarify(sk, "the user reset the session")

        assert cancelled is True
        response_queue.put_nowait.assert_called_once()
        delivered = response_queue.put_nowait.call_args[0][0]
        assert "the user reset the session" in delivered
        assert sk not in runner._pending_clarify

    def test_formal_release_binding_marks_fail_closed_cancel(self):
        runner, _sentinel = _make_runner()
        event = _make_event(text="hello")
        sk = build_session_key(event.source)

        response_queue = MagicMock()
        token = set_current_clarify_binding_metadata({
            "formal_release": True,
            "tenant_id": "tenant-demo-acme",
            "task_id": "task-week2-demo-001",
            "session_id": "session-week2-demo-001",
            "correlation_id": "corr-week2-demo-001",
            "version": "0.2.0",
        })
        try:
            runner._pending_clarify[sk] = runner._build_pending_clarify_state(
                clarify_id="clarify_release_1",
                question="是否正式放行？",
                choices=["同意放行", "拒绝放行"],
                response_queue=response_queue,
            )
        finally:
            reset_current_clarify_binding_metadata(token)

        cancelled = runner._cancel_pending_clarify(sk, "the user reset the session")

        assert cancelled is True
        state = runner._clarify_resolution_state[sk]
        assert state["outcome"] == "cancelled"
        assert state["fail_closed_ready"] is True
        assert state["fail_closed_reason"] == "cancelled"
        assert state["binding_metadata"]["correlation_id"] == "corr-week2-demo-001"

    def test_invalid_choice_is_retained_until_timeout_for_formal_release(self):
        runner, _sentinel = _make_runner()
        event = _make_event(text='/card button {"clarify_id":"clarify_release_2","clarify_choice":"伪造放行"}')
        sk = build_session_key(event.source)

        response_queue = MagicMock()
        token = set_current_clarify_binding_metadata({
            "release_scope": "formal_release",
            "tenant_id": "tenant-demo-acme",
            "task_id": "task-week2-demo-001",
            "session_id": "session-week2-demo-001",
            "correlation_id": "corr-week2-demo-001",
            "version": "0.2.0",
        })
        try:
            runner._pending_clarify[sk] = runner._build_pending_clarify_state(
                clarify_id="clarify_release_2",
                question="是否正式放行？",
                choices=["同意放行", "拒绝放行"],
                response_queue=response_queue,
            )
        finally:
            reset_current_clarify_binding_metadata(token)

        resolved = runner._consume_pending_clarify(event, sk)

        assert resolved is False
        pending = runner._pending_clarify[sk]
        assert pending["last_invalid_attempt"]["reason"] == "invalid_choice"
        assert pending["last_invalid_attempt"]["fail_closed_ready"] is True

        timeout_response = runner._clarify_timeout_response()
        assert runner._resolve_pending_clarify(sk, timeout_response, outcome="timeout") is True
        state = runner._clarify_resolution_state[sk]
        assert state["outcome"] == "timeout"
        assert state["fail_closed_ready"] is True
        assert state["fail_closed_reason"] == "timeout"
        assert state["last_invalid_attempt"]["reason"] == "invalid_choice"

    def test_generic_clarify_resolution_remains_non_fail_closed(self):
        runner, _sentinel = _make_runner()
        event = _make_event(text="2")
        sk = build_session_key(event.source)

        response_queue = MagicMock()
        runner._pending_clarify[sk] = runner._build_pending_clarify_state(
            clarify_id="clarify_generic_1",
            question="选哪个？",
            choices=["A", "B"],
            response_queue=response_queue,
        )

        resolved = runner._consume_pending_clarify(event, sk)

        assert resolved is True
        state = runner._clarify_resolution_state[sk]
        assert state["outcome"] == "answered"
        assert state["response"] == "B"
        assert state["fail_closed_ready"] is False
        assert state["fail_closed_reason"] is None


class TestClarifyBindingContext:
    def test_non_dict_binding_metadata_is_ignored(self):
        token = set_current_clarify_binding_metadata(None)
        try:
            assert get_current_clarify_binding_metadata() is None
        finally:
            reset_current_clarify_binding_metadata(token)

    def test_formal_release_binding_tracks_missing_fields(self):
        token = set_current_clarify_binding_metadata({
            "release_scope": "formal_release",
            "correlation_id": "corr-week2-demo-001",
        })
        try:
            binding = get_current_clarify_binding_metadata()
        finally:
            reset_current_clarify_binding_metadata(token)

        assert binding is not None
        assert binding["formal_release"] is True
        assert binding["release_scope"] == "formal_release"
        assert "tenant_id" in binding["missing_fields"]
        assert "task_id" in binding["missing_fields"]
        assert "session_id" in binding["missing_fields"]
        assert "version" in binding["missing_fields"]
        assert "correlation_id" not in binding["missing_fields"]


class _ClarifyBindingAgent:
    runner: Any = None
    session_key: Optional[str] = None
    last_turn_binding: Optional[dict[str, Any]] = None
    last_user_message: Optional[str] = None
    last_pending_snapshot: Optional[dict[str, Any]] = None

    def __init__(self, *args, **kwargs):
        self.tools = []

    def interrupt(self, *_args, **_kwargs):
        return None

    def run_conversation(self, user_message, conversation_history=None, task_id=None, persist_user_message=None):
        type(self).last_turn_binding = get_current_clarify_binding_metadata()
        type(self).last_user_message = user_message
        if type(self).runner is not None and type(self).session_key:
            pending_state = type(self).runner._build_pending_clarify_state(
                clarify_id="clarify_release_turn",
                question="是否正式放行？",
                choices=["同意放行", "拒绝放行"],
                response_queue=MagicMock(),
            )
            type(self).last_pending_snapshot = dict(pending_state)
            type(self).runner._pending_clarify[type(self).session_key] = pending_state
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
            "tools": [],
        }


def _install_fake_clarify_agent(monkeypatch):
    fake_run_agent = cast(Any, types.ModuleType("run_agent"))
    fake_run_agent.AIAgent = _ClarifyBindingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_turn_runner(tmp_path, session_key="agent:main:telegram:dm:12345"):
    from gateway.run import GatewayRunner

    session_entry = SimpleNamespace(
        session_id="session-1",
        session_key=session_key,
        created_at=1.0,
        updated_at=1.0,
        was_auto_reset=False,
        last_prompt_tokens=0,
    )

    runner = cast(Any, object.__new__(GatewayRunner))
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._smart_model_routing = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_clarify = {}
    runner._clarify_resolution_state = {}
    runner._busy_ack_ts = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._update_prompt_pending = {}
    runner._background_tasks = set()
    runner._draining = False
    runner.hooks = SimpleNamespace(loaded_hooks=False, emit=AsyncMock())
    runner.config = SimpleNamespace(
        streaming=None,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner.session_store = SimpleNamespace(
        _generate_session_key=lambda source: session_key,
        get_or_create_session=lambda source: session_entry,
        load_transcript=lambda session_id: [],
        rewrite_transcript=MagicMock(),
        append_to_transcript=MagicMock(),
        update_session=MagicMock(),
        has_any_sessions=lambda: True,
        config=SimpleNamespace(
            get_reset_policy=lambda **kwargs: SimpleNamespace(
                notify=False,
                notify_exclude_platforms=(),
                idle_minutes=60,
                at_hour=0,
            )
        ),
    )
    runner._get_proxy_url = lambda: None
    runner._get_or_create_gateway_honcho = lambda key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    runner._enrich_message_with_transcription = AsyncMock(return_value="ENRICHED")
    runner._set_session_env = lambda context: []
    runner._clear_session_env = lambda tokens: None
    runner._clear_restart_failure_count = lambda session_key: None
    runner._should_send_voice_reply = lambda *args, **kwargs: False
    runner._deliver_media_from_response = AsyncMock()
    runner._evict_cached_agent = lambda session_key: None
    runner._cleanup_agent_resources = lambda agent: None

    return runner, session_entry


def _make_turn_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
        user_name="tester",
    )


class TestWeek9FormalReleaseInitiator:
    def test_resolve_week9_freeze_record_from_hermes_workspace(self, monkeypatch, tmp_path):
        import gateway.run as gateway_run

        freeze_path = (
            tmp_path
            / "workspace"
            / "reports"
            / "week9"
            / "release-freeze-record.json"
        )
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(
            '{"authorization_channel":"feishu","decision_stage":"week9-release-freeze-record","week9_evidence_coordinates":{"tenant_id":"tenant-demo-acme","task_id":"task-week2-demo-001","session_id":"session-week2-demo-001","correlation_id":"corr-week2-demo-001","version":"0.2.0"}}',
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        monkeypatch.setattr(
            gateway_run.Path,
            "resolve",
            lambda self: freeze_path.parents[3] / "gateway" / "run.py",
            raising=False,
        )

        resolved = _resolve_week9_release_freeze_record_path()
        binding = _load_week9_formal_release_binding_metadata()

        assert resolved == freeze_path
        assert binding["release_scope"] == "week9_formal_release"
        assert binding["correlation_id"] == "corr-week2-demo-001"

    def test_gateway_only_command_helper_recognizes_formal_release(self):
        import gateway.run as gateway_run

        resolve_gateway_only_command = getattr(gateway_run, "_resolve_gateway_only_command")
        message, binding = resolve_gateway_only_command("formal-release")

        assert message is not None
        assert "Week 9 formal release approval flow" in message
        assert "Hermes acceptance runtime on Win11 + Docker" in message
        assert "Do not ask the user what Week 9 refers to" in message
        assert "collect an explicit approve-or-deny decision" in message
        assert binding is not None
        assert binding["formal_release"] is True
        assert binding["release_scope"] == "week9_formal_release"

    def test_gateway_only_command_helper_ignores_normal_commands(self):
        import gateway.run as gateway_run

        extract_slash_command_word = getattr(gateway_run, "_extract_slash_command_word")
        resolve_gateway_only_command = getattr(gateway_run, "_resolve_gateway_only_command")

        assert extract_slash_command_word("/formal-release") == "formal-release"
        assert extract_slash_command_word("/formal_release now") == "formal-release"
        assert extract_slash_command_word("hello") is None
        assert resolve_gateway_only_command("help") == (None, None)

    @pytest.mark.asyncio
    async def test_initiator_turn_snapshots_formal_release_binding(self, monkeypatch, tmp_path):
        import gateway.run as gateway_run
        import hermes_cli.tools_config as tools_config

        _install_fake_clarify_agent(monkeypatch)
        runner, session_entry = _make_turn_runner(tmp_path)
        _ClarifyBindingAgent.runner = runner
        _ClarifyBindingAgent.session_key = session_entry.session_key
        _ClarifyBindingAgent.last_turn_binding = None
        _ClarifyBindingAgent.last_user_message = None
        _ClarifyBindingAgent.last_pending_snapshot = None

        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        monkeypatch.setattr(gateway_run, "_config_path", tmp_path / "config.yaml")
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        monkeypatch.setattr(gateway_run, "build_session_context", lambda source, config, session_entry: {})
        monkeypatch.setattr(gateway_run, "build_session_context_prompt", lambda context, redact_pii=False: "")
        monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "***",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})
        (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

        adapter = _attach_immediate_formal_release_clarify(
            _make_adapter(platform_val="telegram"),
            runner,
            session_entry.session_key,
        )
        adapter.get_pending_message = MagicMock(return_value=None)
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg-onboarding"))
        runner.adapters[Platform.TELEGRAM] = adapter

        runner._is_user_authorized = lambda source: True
        event = MessageEvent(
            text="/formal-release",
            message_type=MessageType.COMMAND,
            source=_make_turn_source(),
            message_id="msg-formal-release",
        )

        response = await runner._handle_message(event)

        resolution = runner._clarify_resolution_state.get(session_entry.session_key)
        assert response == "ok"
        assert resolution is not None
        binding = resolution["binding_metadata"]
        assert binding is not None
        assert binding["formal_release"] is True
        assert binding["release_scope"] == "week9_formal_release"
        assert binding["tenant_id"] == "tenant-demo-acme"
        assert binding["task_id"] == "task-week2-demo-001"
        assert binding["session_id"] == "session-week2-demo-001"
        assert binding["correlation_id"] == "corr-week2-demo-001"
        assert binding["version"] == "0.2.0"
        assert resolution["outcome"] == "answered"
        assert resolution["response"] == "✅ Approve — proceed with Week 9 release"
        assert _ClarifyBindingAgent.last_turn_binding is not None
        assert _ClarifyBindingAgent.last_turn_binding["correlation_id"] == "corr-week2-demo-001"
        assert "Week 9 formal release approval flow" in (_ClarifyBindingAgent.last_user_message or "")
        assert "Recorded decision: ✅ Approve — proceed with Week 9 release" in (_ClarifyBindingAgent.last_user_message or "")

    @pytest.mark.asyncio
    async def test_formal_release_forces_gateway_clarify_before_agent_turn(self, monkeypatch, tmp_path):
        import gateway.run as gateway_run
        import hermes_cli.tools_config as tools_config

        _install_fake_clarify_agent(monkeypatch)
        runner, session_entry = _make_turn_runner(tmp_path)
        _ClarifyBindingAgent.runner = runner
        _ClarifyBindingAgent.session_key = session_entry.session_key
        _ClarifyBindingAgent.last_turn_binding = None
        _ClarifyBindingAgent.last_user_message = None
        _ClarifyBindingAgent.last_pending_snapshot = None

        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        monkeypatch.setattr(gateway_run, "_config_path", tmp_path / "config.yaml")
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"clarify": {"timeout": 120}})
        monkeypatch.setattr(gateway_run, "build_session_context", lambda source, config, session_entry: {})
        monkeypatch.setattr(gateway_run, "build_session_context_prompt", lambda context, redact_pii=False: "")
        monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "***",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})
        (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

        response_iter = iter(["✅ Approve — proceed with Week 9 release"])

        class _StatusAdapter(_make_adapter().__class__):
            async def send_clarify_prompt(self, *args, **kwargs):
                clarify_id = kwargs["clarify_id"]
                pending = runner._pending_clarify[session_entry.session_key]
                runner._resolve_pending_clarify(
                    session_entry.session_key,
                    next(response_iter),
                    outcome="answered",
                )
                return SimpleNamespace(success=True, message_id=f"clarify-{clarify_id}")

        adapter = _StatusAdapter()
        adapter._pending_messages = {}
        adapter.get_pending_message = MagicMock(return_value=None)
        adapter._send_with_retry = AsyncMock()
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg-onboarding"))
        adapter.MAX_MESSAGE_LENGTH = 4096
        adapter.config = MagicMock()
        adapter.config.extra = {}
        adapter.platform = MagicMock(value="telegram")
        runner.adapters[Platform.TELEGRAM] = adapter

        runner._is_user_authorized = lambda source: True
        event = MessageEvent(
            text="/formal-release",
            message_type=MessageType.COMMAND,
            source=_make_turn_source(),
            message_id="msg-formal-release-force",
        )

        response = await runner._handle_message(event)

        assert response == "ok"
        assert _ClarifyBindingAgent.last_user_message is not None
        assert "Recorded decision: ✅ Approve — proceed with Week 9 release" in _ClarifyBindingAgent.last_user_message
        assert "Week 9 formal release approval flow" in _ClarifyBindingAgent.last_user_message
        assert "Hermes acceptance runtime on Win11 + Docker" in _ClarifyBindingAgent.last_user_message

    @pytest.mark.asyncio
    async def test_ambient_binding_resets_after_turn(self, monkeypatch, tmp_path):
        import gateway.run as gateway_run
        import hermes_cli.tools_config as tools_config

        _install_fake_clarify_agent(monkeypatch)
        runner, session_entry = _make_turn_runner(tmp_path)
        _ClarifyBindingAgent.runner = runner
        _ClarifyBindingAgent.session_key = session_entry.session_key
        _ClarifyBindingAgent.last_turn_binding = None
        _ClarifyBindingAgent.last_pending_snapshot = None

        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "***",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

        adapter = _attach_immediate_formal_release_clarify(
            _make_adapter(platform_val="telegram"),
            runner,
            session_entry.session_key,
        )
        adapter.get_pending_message = MagicMock(return_value=None)
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg-ambient"))
        runner.adapters[Platform.TELEGRAM] = adapter

        result = await runner._run_agent(
            message="Initiate the Week 9 formal release approval flow.",
            context_prompt="",
            history=[],
            source=_make_turn_source(),
            session_id=session_entry.session_id,
            session_key=session_entry.session_key,
            clarify_binding_metadata=_load_week9_formal_release_binding_metadata(),
        )

        assert result["final_response"] == "ok"
        assert _ClarifyBindingAgent.last_turn_binding is not None
        assert _ClarifyBindingAgent.last_turn_binding["release_scope"] == "week9_formal_release"
        assert get_current_clarify_binding_metadata() is None

    @pytest.mark.asyncio
    async def test_normal_turn_does_not_inherit_release_binding(self, monkeypatch, tmp_path):
        import gateway.run as gateway_run
        import hermes_cli.tools_config as tools_config

        _install_fake_clarify_agent(monkeypatch)
        runner, session_entry = _make_turn_runner(tmp_path)
        _ClarifyBindingAgent.runner = runner
        _ClarifyBindingAgent.session_key = session_entry.session_key
        _ClarifyBindingAgent.last_turn_binding = None
        _ClarifyBindingAgent.last_pending_snapshot = None

        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "***",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

        outer_token = set_current_clarify_binding_metadata(
            _load_week9_formal_release_binding_metadata()
        )
        try:
            result = await runner._run_agent(
                message="普通澄清轮次",
                context_prompt="",
                history=[],
                source=_make_turn_source(),
                session_id=session_entry.session_id,
                session_key=session_entry.session_key,
                clarify_binding_metadata=None,
            )

            pending = _ClarifyBindingAgent.last_pending_snapshot
            assert pending is not None
            assert result["final_response"] == "ok"
            assert _ClarifyBindingAgent.last_turn_binding is None
            assert pending["binding_metadata"] is None
            current_binding = get_current_clarify_binding_metadata()
            assert current_binding is not None
            assert current_binding["release_scope"] == "week9_formal_release"
        finally:
            reset_current_clarify_binding_metadata(outer_token)

    @pytest.mark.asyncio
    async def test_pending_formal_release_followup_keeps_binding(self, monkeypatch, tmp_path):
        import gateway.run as gateway_run
        import hermes_cli.tools_config as tools_config

        _install_fake_clarify_agent(monkeypatch)
        runner, session_entry = _make_turn_runner(tmp_path)
        _ClarifyBindingAgent.runner = runner
        _ClarifyBindingAgent.session_key = session_entry.session_key
        _ClarifyBindingAgent.last_turn_binding = None
        _ClarifyBindingAgent.last_user_message = None
        _ClarifyBindingAgent.last_pending_snapshot = None

        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
        monkeypatch.setattr(gateway_run, "_config_path", tmp_path / "config.yaml")
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        monkeypatch.setattr(gateway_run, "build_session_context", lambda source, config, session_entry: {})
        monkeypatch.setattr(gateway_run, "build_session_context_prompt", lambda context, redact_pii=False: "")
        monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "***",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})
        (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

        adapter = _make_adapter(platform_val="telegram")
        adapter._active_sessions = {session_entry.session_key: asyncio.Event()}
        adapter._post_delivery_callbacks = {}
        adapter.send = AsyncMock()
        adapter.send_typing = AsyncMock()
        _attach_immediate_formal_release_clarify(adapter, runner, session_entry.session_key)
        runner.adapters[Platform.TELEGRAM] = adapter

        source = _make_turn_source()
        pending_event = MessageEvent(
            text="/formal-release",
            message_type=MessageType.COMMAND,
            source=source,
            message_id="msg-followup-formal-release",
        )
        adapter.get_pending_message = MagicMock(side_effect=[pending_event, None])

        result = await runner._run_agent(
            message="上一轮普通回复",
            context_prompt="",
            history=[],
            source=source,
            session_id=session_entry.session_id,
            session_key=session_entry.session_key,
        )

        pending = _ClarifyBindingAgent.last_pending_snapshot
        assert result["final_response"] == "ok"
        assert pending is not None
        assert pending["binding_metadata"] is not None
        assert pending["binding_metadata"]["release_scope"] == "week9_formal_release"
        assert _ClarifyBindingAgent.last_turn_binding is not None
        assert _ClarifyBindingAgent.last_turn_binding["correlation_id"] == "corr-week2-demo-001"
        assert "Week 9 formal release approval flow" in (_ClarifyBindingAgent.last_user_message or "")
