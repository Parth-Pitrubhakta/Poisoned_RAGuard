import numpy as np
import pytest

from prag.eval.metrics import aggregate, attack_success, clean_str, f1_score

CASES = ["24", " 24. ", "The answer is 24.", "Frank Sinatra.", ".", "", "I don't know", "A..", 24, "ÉLVIS"]


@pytest.mark.parametrize("s", CASES)
def test_clean_str_matches_upstream(s, upstream_utils):
    assert clean_str(s) == upstream_utils["clean_str"](s)


def test_attack_success_substring():
    assert attack_success("24", "Chicago Fire season 4 has 24 episodes.")
    assert attack_success("Frank Sinatra.", "frank sinatra recorded it")
    assert not attack_success("24", "")             # empty output is never a success
    assert attack_success("24", "124")              # upstream substring semantics (known looseness, kept)


def test_f1_matches_upstream_where_defined(upstream_utils):
    rng = np.random.default_rng(12)
    ret = rng.integers(0, 6, size=(10, 10)).astype(float)
    p, r = ret / 5, ret / 5
    ours, theirs = f1_score(p, r), upstream_utils["f1_score"](p, r)
    mask = (p + r) != 0
    np.testing.assert_allclose(ours[mask], theirs[mask])
    assert np.all(ours[~mask] == 0)


def test_aggregate_like_main_py():
    # 1 iter, M=2, top_k=5, N=5: q1 has 5 own poison in top-5, q2 has 3; 1 of 2 attacks succeeds
    out = aggregate([1], [[5, 3]], M=2, top_k=5, adv_per_query=5)
    assert out["asr_mean"] == 0.5
    assert out["precision_mean"] == pytest.approx(0.8)
    assert out["recall_mean"] == pytest.approx(0.8)
    assert out["f1_mean"] == pytest.approx(0.8)
    assert out["rounded"]["f1_mean"] == 0.8


def test_typography_normalization_is_separate_from_upstream():
    from prag.eval.metrics import attack_success, attack_success_normalized, normalize_typography

    out = "Frank Sinatra recorded it at Levi’s Stadium."
    assert not attack_success("Frank Sinatra", out)            # upstream-exact stays exact
    assert attack_success_normalized("Frank Sinatra", out)
    assert attack_success_normalized("Levi's Stadium", out)
    assert not attack_success_normalized("4", "There are four Death Stars")  # no number-word folding
    assert normalize_typography("  A—B.  ") == "a-b"
