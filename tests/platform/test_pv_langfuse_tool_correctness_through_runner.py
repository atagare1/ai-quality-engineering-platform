"""PV: real Langfuse SUT trajectory through ToolCorrectness and QualityGate.

Lifecycle: RUN SCENARIO → COMPLETE AGENT EXECUTION → FLUSH → SETTLE/PULL
TRACE → EXTRACT TOOL CALLS → request map → EvaluationRunner →
ToolCorrectness → Policy → Gate.

External System Under Test: spikes/langfuse_openai_agents (subprocess).
ai_qe_eval only retrieves Langfuse observations and evaluates them.

Does not construct EvaluationTrace. Final-state tests compute a bool from
extracted ToolInvocation results and pass a request map to Runner.

Does not import the OpenAI Agents SDK. Final agent text is a QE specification
string, not Runner.final_output.

Order tests use evaluation_params=[]. The clock-argument test uses
INPUT_PARAMETERS only and does not score dynamic tool output.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from deepeval.metrics.tool_correctness.tool_correctness import ToolCorrectnessMetric
from deepeval.test_case import ToolCallParams
from deepeval_live import live_deepeval_local_model
from dotenv import load_dotenv

from ai_qe_eval.capture.langfuse_trace import (
    langfuse_tool_correctness_request,
    observed_tool_invocations,
)
from ai_qe_eval.domain.config import EvaluationConfig
from ai_qe_eval.domain.conversation import ToolInvocation
from ai_qe_eval.domain.registry import EvaluationCapability, EvaluationRegistry
from ai_qe_eval.evaluators.deepeval_tool_correctness import DeepEvalToolCorrectnessEvaluator
from ai_qe_eval.evaluators.deterministic import FINAL_STATE_METRIC, FinalStateEvaluator
from ai_qe_eval.gate.quality_gate import QualityGate
from ai_qe_eval.integrations.langfuse_observations import (
    LangfuseTraceIncompleteError,
    flush_langfuse_client,
    wait_for_settled_observations,
)
from ai_qe_eval.policy.quality_policy import QualityPolicy
from ai_qe_eval.runner.evaluation_runner import EvaluationRunner

PLATFORM_ROOT = Path(__file__).resolve().parents[2]
SPIKE_DIR = PLATFORM_ROOT / "spikes" / "langfuse_openai_agents"
SPIKE_SCRIPT = SPIKE_DIR / "run_spike.py"
TRACE_ID_PATH = SPIKE_DIR / "artifacts" / "last_trace_id.txt"
SHORT_VENV_PYTHON = Path(r"C:\Users\aarya\.venvs\lf-oa-spike\Scripts\python.exe")
LOCAL_SPIKE_PYTHON = SPIKE_DIR / ".venv" / "Scripts" / "python.exe"

SCENARIO_TOOL_NAMES = [
    "fetch_timezone_clock",
    "fetch_public_uuid",
    "fetch_httpbin_json",
]
EXPECTED_TOOL_ORDER = list(SCENARIO_TOOL_NAMES)
WRONG_TOOL_ORDER = [
    "fetch_httpbin_json",
    "fetch_timezone_clock",
    "fetch_public_uuid",
]
SPEC_INPUT = (
    "Collect UTC time, a public UUID, and httpbin json, then summarize all three."
)
SPEC_OUTPUT = "QE specification: final agent text is not ingested from Langfuse."
CLOCK_TOOL_NAME = "fetch_timezone_clock"
CLOCK_EXPECTED_ARGUMENTS = {"iana_timezone": "UTC"}
HTTPBIN_TOOL_NAME = "fetch_httpbin_json"
QE_EXPECTED_SLIDESHOW_TITLE = "Sample Slide Show"
WRONG_SLIDESHOW_TITLE = "Not A Real Slideshow"
TOOL_CORRECTNESS_THRESHOLD = 0.80


def _spike_python() -> Path:
    if SHORT_VENV_PYTHON.exists():
        return SHORT_VENV_PYTHON
    if LOCAL_SPIKE_PYTHON.exists():
        return LOCAL_SPIKE_PYTHON
    return Path(sys.executable)


def _require_live_env() -> None:
    load_dotenv(PLATFORM_ROOT / ".env")
    required = (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "OPENROUTER_API_KEY",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.skip("Missing live credentials: " + ", ".join(missing))


def _names_only(names: list[str]) -> list[ToolInvocation]:
    return [ToolInvocation(name=name, arguments=None, result=None) for name in names]


def _captured_result_contains(result, expected_text: str) -> bool:
    if result is None or not expected_text:
        return False
    if isinstance(result, str):
        return expected_text in result
    try:
        return expected_text in json.dumps(result)
    except TypeError:
        return expected_text in str(result)


def _httpbin_result_contains(observed, expected_text: str) -> bool:
    for call in reversed(observed or []):
        if call.name == HTTPBIN_TOOL_NAME:
            return _captured_result_contains(call.result, expected_text)
    return False


def _scenario_trace_complete(rows) -> bool:
    names = [
        str(row.get("name") or "")
        for row in rows
        if str(row.get("type") or "").upper() == "TOOL"
    ]
    return len(names) >= len(SCENARIO_TOOL_NAMES) and all(
        name in names for name in SCENARIO_TOOL_NAMES
    )


def _order_evaluator() -> DeepEvalToolCorrectnessEvaluator:
    metric = ToolCorrectnessMetric(
        should_exact_match=True,
        available_tools=None,
        evaluation_params=[],
        include_reason=True,
        async_mode=False,
        model=live_deepeval_local_model(),
    )
    return DeepEvalToolCorrectnessEvaluator(
        tool_correctness_metric=metric,
        evaluation_params=[],
    )


def _input_parameters_evaluator() -> DeepEvalToolCorrectnessEvaluator:
    metric = ToolCorrectnessMetric(
        should_exact_match=True,
        available_tools=None,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        include_reason=True,
        async_mode=False,
        model=live_deepeval_local_model(),
    )
    return DeepEvalToolCorrectnessEvaluator(
        tool_correctness_metric=metric,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
    )


def _tool_correctness_runner(evaluator) -> EvaluationRunner:
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name="tool_correctness",
            evaluator="deepeval",
            category="agent",
        )
    )
    return EvaluationRunner(
        registry=registry,
        evaluators={"tool_correctness": evaluator},
        policies={
            "tool_correctness": QualityPolicy(
                metric="tool_correctness",
                operator=">=",
                threshold=TOOL_CORRECTNESS_THRESHOLD,
            )
        },
        gate=QualityGate(),
    )


def _final_state_runner() -> EvaluationRunner:
    registry = EvaluationRegistry()
    registry.register(
        EvaluationCapability(
            name=FINAL_STATE_METRIC,
            evaluator="deterministic",
            category="agent",
        )
    )
    return EvaluationRunner(
        registry=registry,
        evaluators={
            FINAL_STATE_METRIC: FinalStateEvaluator(),
        },
        policies={
            FINAL_STATE_METRIC: QualityPolicy(
                metric=FINAL_STATE_METRIC,
                operator="==",
                threshold=1.0,
            )
        },
        gate=QualityGate(),
    )


@pytest.fixture(scope="module")
def real_langfuse_observations():
    _require_live_env()
    try:
        from langfuse import get_client
    except ImportError:
        pytest.skip("langfuse is not installed in this environment")
    if not SPIKE_SCRIPT.exists():
        pytest.skip(f"spike script not found: {SPIKE_SCRIPT}")

    python_exe = _spike_python()
    started_at = datetime.now(timezone.utc)
    print("lifecycle RUN_SCENARIO")
    completed = subprocess.run(
        [str(python_exe), str(SPIKE_SCRIPT)],
        cwd=str(SPIKE_DIR),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    print("lifecycle COMPLETE_AGENT_EXECUTION", completed.returncode)
    if completed.returncode != 0:
        pytest.fail(
            "External Langfuse spike SUT failed.\n"
            f"stdout:\n{completed.stdout[-2000:]}\n"
            f"stderr:\n{completed.stderr[-2000:]}"
        )
    trace_id = TRACE_ID_PATH.read_text(encoding="utf-8").strip()
    if not trace_id:
        pytest.fail("trace_id missing after scenario completion")
    print("langfuse_trace_id", trace_id)
    client = get_client()
    if not client.auth_check():
        pytest.fail("Langfuse auth_check() failed")
    flushed = flush_langfuse_client(client)
    print("lifecycle FLUSH", flushed)
    try:
        observations = wait_for_settled_observations(
            client,
            trace_id,
            from_start_time=started_at - timedelta(minutes=5),
            is_complete=_scenario_trace_complete,
            timeout_s=45.0,
            poll_interval_s=0.5,
            page_limit=2,
        )
    except LangfuseTraceIncompleteError as exc:
        pytest.fail(f"Langfuse trace did not settle: {exc}")
    print("lifecycle PULL_SETTLED_TRACE", len(observations))
    print("qe_supplied_input", SPEC_INPUT)
    print("qe_supplied_output", SPEC_OUTPUT)
    observed = observed_tool_invocations(observations, trace_id=trace_id)
    observed_names = [call.name for call in observed]
    print("observed_tool_names", observed_names)
    print("observation_count", len(observations))
    if observed_names != SCENARIO_TOOL_NAMES:
        pytest.fail(
            "SUT did not produce the specified A→B→C tool order. "
            f"observed={observed_names} expected={SCENARIO_TOOL_NAMES}"
        )
    return {
        "trace_id": trace_id,
        "observations": observations,
        "observed": observed,
    }


@pytest.mark.live
def test_pv_langfuse_correct_trajectory_tool_correctness_gate_pass(
    real_langfuse_observations,
):
    payload = real_langfuse_observations
    observed = payload["observed"]
    expected = _names_only(EXPECTED_TOOL_ORDER)
    print("lifecycle EXTRACT_TOOL_CALLS")
    assert [call.name for call in observed] == EXPECTED_TOOL_ORDER
    assert [call.name for call in expected] == EXPECTED_TOOL_ORDER
    assert all(call.arguments is None for call in expected)

    print("lifecycle EVALUATE")
    request = langfuse_tool_correctness_request(
        observations=payload["observations"],
        trace_id=payload["trace_id"],
        expected_tool_calls=expected,
    )
    runner = _tool_correctness_runner(_order_evaluator())
    gate = runner.run(
        request,
        EvaluationConfig(evaluations=["tool_correctness"]),
        run_id="pv-langfuse-correct-trajectory",
    )
    assert runner.last_run is not None
    result = runner.last_run.results[0]
    policy_decision = runner.last_run.decisions[0]
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("gate_passed", gate.passed)
    assert result.metric == "tool_correctness"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert result.score >= TOOL_CORRECTNESS_THRESHOLD
    assert gate.passed is True
    assert policy_decision.passed is True


@pytest.mark.live
def test_pv_langfuse_incorrect_trajectory_tool_correctness_gate_fail(
    real_langfuse_observations,
):
    payload = real_langfuse_observations
    observed = payload["observed"]
    expected = _names_only(WRONG_TOOL_ORDER)
    print("lifecycle EXTRACT_TOOL_CALLS")
    assert [call.name for call in observed] == EXPECTED_TOOL_ORDER
    assert [call.name for call in expected] == WRONG_TOOL_ORDER

    print("lifecycle EVALUATE")
    request = langfuse_tool_correctness_request(
        observations=payload["observations"],
        trace_id=payload["trace_id"],
        expected_tool_calls=expected,
    )
    runner = _tool_correctness_runner(_order_evaluator())
    gate = runner.run(
        request,
        EvaluationConfig(evaluations=["tool_correctness"]),
        run_id="pv-langfuse-incorrect-trajectory",
    )
    assert runner.last_run is not None
    result = runner.last_run.results[0]
    policy_decision = runner.last_run.decisions[0]
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("gate_passed", gate.passed)
    assert result.metric == "tool_correctness"
    assert gate.passed is False
    assert policy_decision.passed is False
    assert result.score < TOOL_CORRECTNESS_THRESHOLD


@pytest.mark.live
def test_pv_langfuse_clock_argument_tool_correctness_gate_pass(
    real_langfuse_observations,
):
    payload = real_langfuse_observations
    clock_observations = [
        row
        for row in payload["observations"]
        if str(row.get("type") or "").upper() == "TOOL"
        and row.get("name") == CLOCK_TOOL_NAME
    ]
    assert len(clock_observations) == 1
    print("lifecycle EXTRACT_TOOL_CALLS")
    observed_clock = observed_tool_invocations(
        clock_observations, trace_id=payload["trace_id"]
    )
    assert observed_clock[0].arguments == CLOCK_EXPECTED_ARGUMENTS

    expected_clock = ToolInvocation(
        name=CLOCK_TOOL_NAME,
        arguments={"iana_timezone": "UTC"},
        result=None,
    )
    assert [call.name for call in observed_clock] == [CLOCK_TOOL_NAME]
    assert [call.name for call in [expected_clock]] == [CLOCK_TOOL_NAME]
    assert expected_clock.arguments == {"iana_timezone": "UTC"}
    assert expected_clock.result is None
    assert expected_clock is not observed_clock[0]

    print("lifecycle EVALUATE")
    print("evaluation_params", ["INPUT_PARAMETERS"])
    request = langfuse_tool_correctness_request(
        observations=clock_observations,
        trace_id=payload["trace_id"],
        expected_tool_calls=[expected_clock],
    )
    runner = _tool_correctness_runner(_input_parameters_evaluator())
    gate = runner.run(
        request,
        EvaluationConfig(evaluations=["tool_correctness"]),
        run_id="pv-langfuse-clock-argument",
    )
    assert runner.last_run is not None
    result = runner.last_run.results[0]
    policy_decision = runner.last_run.decisions[0]
    print("tool_correctness_score", result.score)
    print("tool_correctness_reason", result.reason)
    print("gate_passed", gate.passed)
    assert result.metric == "tool_correctness"
    assert isinstance(result.score, (int, float)) and not isinstance(result.score, bool)
    assert math.isfinite(result.score)
    assert result.score >= TOOL_CORRECTNESS_THRESHOLD
    assert gate.passed is True
    assert policy_decision.passed is True


def _require_httpbin_slideshow_evidence(payload):
    httpbin = [call for call in payload["observed"] if call.name == HTTPBIN_TOOL_NAME]
    assert len(httpbin) == 1, f"expected one {HTTPBIN_TOOL_NAME} TOOL, got {len(httpbin)}"
    captured = httpbin[0].result
    print("httpbin_result_type", type(captured).__name__)
    print(
        "httpbin_result_contains_qe_expected",
        _captured_result_contains(captured, QE_EXPECTED_SLIDESHOW_TITLE),
    )
    if not _captured_result_contains(captured, QE_EXPECTED_SLIDESHOW_TITLE):
        pytest.fail(
            "Fresh Langfuse TOOL result for fetch_httpbin_json does not contain "
            f"{QE_EXPECTED_SLIDESHOW_TITLE!r}; captured={captured!r}"
        )
    return captured


@pytest.mark.live
def test_pv_langfuse_httpbin_final_state_gate_pass(real_langfuse_observations):
    payload = real_langfuse_observations
    captured = _require_httpbin_slideshow_evidence(payload)
    print("lifecycle EXTRACT_TOOL_CALLS")
    assert _captured_result_contains(captured, QE_EXPECTED_SLIDESHOW_TITLE)
    assert _httpbin_result_contains(payload["observed"], QE_EXPECTED_SLIDESHOW_TITLE)

    runner = _final_state_runner()
    print("lifecycle EVALUATE")
    decision = runner.run(
        {
            FINAL_STATE_METRIC: {
                "args": [
                    _httpbin_result_contains(
                        payload["observed"], QE_EXPECTED_SLIDESHOW_TITLE
                    )
                ],
                "kwargs": {},
            }
        },
        EvaluationConfig(evaluations=[FINAL_STATE_METRIC]),
        run_id="pv-langfuse-httpbin-final-state-pass",
    )
    result = runner.last_run.results[0]
    print("final_state_score", result.score)
    print("final_state_reason", result.reason)
    print("gate_passed", decision.passed)
    assert result.metric == FINAL_STATE_METRIC
    assert result.score == 1.0
    assert decision.passed is True
    assert runner.last_run.decisions[0].passed is True


@pytest.mark.live
def test_pv_langfuse_httpbin_final_state_wrong_expected_gate_fail(
    real_langfuse_observations,
):
    payload = real_langfuse_observations
    _require_httpbin_slideshow_evidence(payload)
    print("lifecycle EXTRACT_TOOL_CALLS")
    assert _httpbin_result_contains(payload["observed"], QE_EXPECTED_SLIDESHOW_TITLE)
    assert not _httpbin_result_contains(payload["observed"], WRONG_SLIDESHOW_TITLE)

    runner = _final_state_runner()
    print("lifecycle EVALUATE")
    decision = runner.run(
        {
            FINAL_STATE_METRIC: {
                "args": [
                    _httpbin_result_contains(payload["observed"], WRONG_SLIDESHOW_TITLE)
                ],
                "kwargs": {},
            }
        },
        EvaluationConfig(evaluations=[FINAL_STATE_METRIC]),
        run_id="pv-langfuse-httpbin-final-state-fail",
    )
    result = runner.last_run.results[0]
    print("final_state_score", result.score)
    print("final_state_reason", result.reason)
    print("gate_passed", decision.passed)
    assert result.metric == FINAL_STATE_METRIC
    assert result.score == 0.0
    assert decision.passed is False
    assert runner.last_run.decisions[0].passed is False
