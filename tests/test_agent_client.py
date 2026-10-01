"""House request budgets at the client boundary; no external model calls."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import openai
import pytest

from agent import client as client_module
from agent.client import HouseModelClient


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_house(monkeypatch):
    clock = FakeClock()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")
            ]
        )

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(client_module, "OpenAI", lambda **kwargs: sdk)
    monkeypatch.setattr(client_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(client_module.time, "sleep", clock.sleep)
    monkeypatch.setenv("HOUSE_REASONING", "off")
    monkeypatch.delenv("AGENT_TRANSCRIPT_DIR", raising=False)
    client = HouseModelClient(base_url="http://house.invalid", api_key="fake", model_name="fake")
    return client, clock, sdk, calls


MESSAGES = [{"role": "user", "content": "Solve the task"}]


@pytest.mark.parametrize("phase", ["tests", "audit", "compact"])
def test_code_phases_reserve_reply_without_changing_generation(fake_house, phase):
    client, _, _, calls = fake_house
    client.reasoning_mode = "low"
    client.chat(MESSAGES, phase=phase)
    client.chat(MESSAGES, phase="generate")
    assert calls[0]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert calls[1]["extra_body"]["chat_template_kwargs"] == {"low_effort": True}
    assert client.reasoning_mode == "low"


def test_expired_deadline_does_not_admit_a_request(fake_house):
    client, clock, _, calls = fake_house

    with pytest.raises(TimeoutError, match="deadline"):
        client.chat(MESSAGES, deadline=clock.now)

    assert client.request_count == 0
    assert calls == []


def test_request_timeout_is_clipped_to_remaining_unit_time(fake_house):
    client, clock, _, calls = fake_house

    result = client.chat(MESSAGES, deadline=clock.now + 7.5)

    assert result.content == "answer"
    assert calls[0]["timeout"] == 7.5
    assert "deadline" not in calls[0]
    assert client.request_count == 1


@pytest.mark.parametrize("timeout, expected", [(2.0, 2.0), (30.0, 7.5), (None, 7.5)])
def test_deadline_preserves_a_stricter_caller_timeout(fake_house, timeout, expected):
    client, clock, _, calls = fake_house

    client.chat(MESSAGES, deadline=clock.now + 7.5, timeout=timeout)

    assert calls[0]["timeout"] == expected


def test_deadline_clips_each_phase_of_an_sdk_timeout(fake_house):
    client, clock, _, calls = fake_house
    timeout = openai.Timeout(connect=100.0, read=2.0, write=None, pool=5.0)

    client.chat(MESSAGES, deadline=clock.now + 7.5, timeout=timeout)

    assert calls[0]["timeout"].as_dict() == {
        "connect": 7.5,
        "read": 2.0,
        "write": 7.5,
        "pool": 5.0,
    }
    assert timeout.as_dict() == {"connect": 100.0, "read": 2.0, "write": None, "pool": 5.0}


@pytest.mark.parametrize("request_seconds", [9.0, 10.0])
def test_transient_error_does_not_wait_past_the_deadline(fake_house, request_seconds):
    client, clock, sdk, calls = fake_house

    def disconnect(**kwargs):
        calls.append(kwargs)
        clock.now += request_seconds
        raise openai.APIConnectionError(request=SimpleNamespace())

    sdk.chat.completions.create = disconnect

    with pytest.raises(TimeoutError, match="deadline"):
        client.chat(MESSAGES, deadline=clock.now + 10.0)

    assert client.request_count == 1
    assert len(calls) == 1
    assert clock.sleeps == []


def test_retry_timeout_uses_the_time_left_after_request_and_backoff(fake_house):
    client, clock, sdk, calls = fake_house
    respond = sdk.chat.completions.create

    def disconnect_once(**kwargs):
        if not calls:
            calls.append(kwargs)
            clock.now += 3.0
            raise openai.APIConnectionError(request=SimpleNamespace())
        return respond(**kwargs)

    sdk.chat.completions.create = disconnect_once

    result = client.chat(MESSAGES, deadline=clock.now + 10.0)

    assert result.content == "answer"
    assert [call["timeout"] for call in calls] == [10.0, 5.0]
    assert clock.sleeps == [2.0]
    assert client.request_count == 2


def test_without_a_deadline_transient_retries_keep_the_existing_limits(fake_house):
    client, clock, sdk, calls = fake_house
    respond = sdk.chat.completions.create

    def disconnect_twice(**kwargs):
        if len(calls) < 2:
            calls.append(kwargs)
            raise openai.APIConnectionError(request=SimpleNamespace())
        return respond(**kwargs)

    sdk.chat.completions.create = disconnect_twice

    result = client.chat(MESSAGES, timeout=2.0)

    assert result.content == "answer"
    assert clock.sleeps == [2.0, 8.0]
    assert client.request_count == 3
    assert [call["timeout"] for call in calls] == [2.0, 2.0, 2.0]
    assert all("deadline" not in call for call in calls)


def test_retries_do_not_exceed_the_remaining_request_allowance(fake_house):
    client, clock, sdk, calls = fake_house
    client.max_requests = 1

    def disconnect(**kwargs):
        calls.append(kwargs)
        raise openai.APIConnectionError(request=SimpleNamespace())

    sdk.chat.completions.create = disconnect

    with pytest.raises(openai.APIConnectionError):
        client.chat(MESSAGES, deadline=clock.now + 100.0)

    assert client.request_count == 1
    assert len(calls) == 1
    assert clock.sleeps == []


def test_concurrent_deadline_requests_respect_the_hard_admission_limit(fake_house):
    client, clock, _, calls = fake_house

    def query(_):
        try:
            client.chat(MESSAGES, deadline=clock.now + 10.0, max_tokens=10000)
            return True
        except RuntimeError:
            return False

    with ThreadPoolExecutor(max_workers=8) as workers:
        completed = list(workers.map(query, range(40)))

    assert sum(completed) == 25
    assert client.request_count == 25
    assert len(calls) == 25
    assert all(call["timeout"] == 10.0 and call["max_tokens"] == 4000 for call in calls)
