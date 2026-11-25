import json

import pytest

from prag.attack.lm_targeted import lm_targeted_poison
from prag.data.beir import contriever_index_text, corpus_text, load_corpus, load_qrels, load_queries
from prag.data.targets import TargetQuestion, load_targets
from prag.retrieval.merge import Passage, count_own_poison, poisoned_topk

from conftest import UPSTREAM


@pytest.fixture
def tiny_beir(tmp_path):
    (tmp_path / "qrels").mkdir()
    docs = [{"_id": "d1", "title": "T1", "text": "alpha"}, {"_id": "d2", "title": "", "text": "beta"},
            {"_id": "d3", "title": "T3", "text": "gamma"}]
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs) + "\n")
    (tmp_path / "queries.jsonl").write_text('{"_id": "q1", "text": "what?"}\n{"_id": "q2", "text": "unused"}\n')
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")
    return tmp_path


def test_beir_loaders(tiny_beir):
    c = load_corpus(tiny_beir)
    assert c["d2"] == {"title": "", "text": "beta"}
    assert load_corpus(tiny_beir, ids=["d3"]) == {"d3": {"title": "T3", "text": "gamma"}}
    with pytest.raises(KeyError):
        load_corpus(tiny_beir, ids=["nope"])
    assert load_queries(tiny_beir, "test") == {"q1": "what?"}   # filtered by qrels like GenericDataLoader
    assert load_qrels(tiny_beir, "test") == {"q1": {"d1": 1}}
    assert corpus_text(c["d1"]) == "alpha"                       # RAG context: text only
    assert contriever_index_text(c["d1"]) == "T1 alpha"          # upstream index encoding
    assert contriever_index_text(c["d2"]) == "beta"


def test_released_targets_load():
    ts = load_targets(UPSTREAM / "results" / "adv_targeted_results" / "nq.json")
    assert len(ts) == 100 and ts[0].id == "test1" and len(ts[0].adv_texts) == 5
    assert ts[0].incorrect_answer == "24" and ts[0].correct_answer == "23"


def test_lm_targeted_poison_format():
    t = TargetQuestion("q", "who?", "A", "B", ("B did it.", "It was B."))
    assert lm_targeted_poison(t, 2) == ["who?.B did it.", "who?.It was B."]
    with pytest.raises(ValueError):
        lm_targeted_poison(t, 3)


def test_poisoned_topk_stable_ties_and_ranks():
    clean = [Passage("c1", 1.0, "d1"), Passage("c2", 0.9, "d2"), Passage("c3", 0.8, "d3")]
    poison = [Passage("p1", 1.0, is_poison=True, poison_for="q"), Passage("p2", 0.95, is_poison=True, poison_for="q"),
              Passage("px", 2.0, is_poison=True, poison_for="other")]
    top = poisoned_topk(clean, poison, 3)
    assert [p.text for p in top] == ["px", "c1", "p1"]   # tie at 1.0: clean (appended first) stays ahead
    assert [p.rank for p in top] == [1, 2, 3]
    assert count_own_poison(top, ["p1", "p2"]) == 1       # other question's poison not counted
