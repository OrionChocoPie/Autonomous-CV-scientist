import pytest

from cv_scientist.memory import ResearchMemory
from cv_scientist.models import Experiment, Hypothesis
from cv_scientist.query import ProjectQueryService
from cv_scientist.retrieval_eval import CASES, seed_memory, score_retrieval


@pytest.fixture
def memory(tmp_path):
    memory = ResearchMemory(str(tmp_path / "memory.db"))
    seed_memory(memory)
    yield memory
    memory.close()


@pytest.mark.parametrize("name,query,group,expected", CASES)
def test_fixed_retrieval_evidence(memory, name, query, group, expected):
    report = score_retrieval(memory, [(name, query, group, expected)])
    assert report["passed"] == 1, report["cases"]


def test_no_match_is_not_fabricated_evidence(memory):
    answer = ProjectQueryService(memory).ask("unicorns astronomy zebras")
    assert answer.evidence == [] and "No relevant" in answer.text


@pytest.mark.parametrize("query,limit", [(None, 3), ([], 3), ("small", 0), ("small", -1), ("small", True), ("small", None)])
def test_invalid_search_arguments(memory, query, limit):
    with pytest.raises(ValueError): memory.search(query, limit)


def test_search_index_updates_and_rolls_back(memory):
    memory.add_hypothesis(Hypothesis(id="H-new", statement="Unique zygomatic token"))
    assert memory.search("zygomatic")["hypotheses"][0].id == "H-new"
    memory.add_hypothesis(Hypothesis(id="H-new", statement="Completely replaced"))
    assert memory.search("zygomatic")["hypotheses"] == []
    with pytest.raises(RuntimeError):
        with memory.transaction():
            memory.add_experiment(Experiment(id="rollback", name="zygomatic"))
            raise RuntimeError("rollback")
    assert memory.search("zygomatic")["experiments"] == []


def test_quotes_and_fts_operators_are_treated_as_text(memory):
    assert memory.search('" NOT OR * ) E1')["experiments"][0].id == "E1"
