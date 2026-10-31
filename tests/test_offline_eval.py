import pytest

from cv_scientist.offline_eval import evaluate_offline


@pytest.fixture(scope="module")
def evaluation(): return evaluate_offline()


@pytest.mark.parametrize("scenario", ["metric_tradeoff", "contradictory_counts", "insufficient_evidence", "missing_artifact",
    "unknown_tool", "invalid_arguments", "fabricated_citation", "wrong_evidence_kind", "uncited_retrieved_facts",
    "duplicate_tool_ids", "step_limit", "tool_limit", "token_limit", "unknown_token_usage", "image_limit", "unreachable_goal"])
def test_offline_agent_scenario(evaluation, scenario):
    case = next(row for row in evaluation["cases"] if row["name"] == scenario)
    assert case["passed"], case
    assert evaluation["paid_api_calls"] == 0


def test_offline_report_is_reproducible(evaluation):
    assert evaluate_offline() == evaluation
