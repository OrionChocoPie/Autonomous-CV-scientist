import json
import math

import pytest

from cv_scientist.agent_output import response_json
from cv_scientist.diagnostics import find_regressions
from cv_scientist.models import Experiment


@pytest.mark.parametrize("scope", ["slices", "per_class"])
@pytest.mark.parametrize("metric", ["fp", "fn"])
@pytest.mark.parametrize("count", [0, 1, 10])
def test_count_regressions_from_zero(scope, metric, count):
    baseline = Experiment(id="B0", **{scope: {"small": {metric: 0}}})
    current = Experiment(id="E1", **{scope: {"small": {metric: count}}})
    findings = find_regressions(current, baseline)
    if count == 0:
        assert findings == []
        return
    assert len(findings) == 1
    finding = findings[0]
    assert finding.metric == metric
    assert finding.previous_value == 0 and finding.current_value == count
    assert finding.absolute_delta == count
    assert math.isfinite(finding.relative_delta) and finding.relative_delta == count
    assert "one object" in finding.evidence[0]["detail"]
    assert json.loads(response_json("regressions", {"regressions": findings}))["ok"]


@pytest.mark.parametrize("count,expected", [(99, 0), (100, 0), (101, 0), (102, 1)])
def test_nonzero_count_regression_threshold_is_preserved(count, expected):
    baseline = Experiment(id="B0", slices={"small": {"fp": 100}})
    current = Experiment(id="E1", slices={"small": {"fp": count}})
    assert len(find_regressions(current, baseline)) == expected


def test_missing_count_is_not_inferred_as_zero():
    baseline = Experiment(id="B0", slices={"small": {"fn": 0}})
    current = Experiment(id="E1", slices={"small": {"fp": 10}})
    assert find_regressions(current, baseline) == []
