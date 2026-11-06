import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
import numpy as np
import data as D, run as R

L = D.leads()


def test_funnel_is_monotonic():
    S = R.bind_book(L)
    _, f = R.unit_economics(L, S, .8)
    assert f["leads"] >= f["contacted"] >= f["applied"] >= f["quoted"] >= f["bound"]


def test_appetite_routing_beats_random():
    a, b, _ = R.routing(L)
    assert b["quote_rate"] > a["quote_rate"] and b["bind_rate"] > a["bind_rate"]


def test_faster_response_never_hurts():
    _, s = R.speed_to_lead(L)
    assert s["bound_ai_first_response"] >= s["bound_now"]


def test_claims_guardrail_catches_every_claim():
    t = R.triage(D.inbox())
    assert t["familiar_wording"]["claim_recall_with_guardrail_pct"] == 100.0


def test_drift_eval_is_harder_than_random_split():
    t = R.triage(D.inbox())
    assert t["new_wording"]["model_accuracy"] < t["familiar_wording"]["model_accuracy"] - 30
