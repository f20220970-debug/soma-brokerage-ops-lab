"""Soma Brokerage Ops Lab: five Founder's Office analyses on a simulated US small-business insurance brokerage.

1. Metric tree and unit economics by channel (CAC per bound policy, commission LTV, payback)
2. Speed-to-lead: what an always-on AI first response is worth (after-hours leads wait hours today)
3. Carrier routing: send each application to carriers by appetite instead of at random
4. Renewals: effect of a proactive renewal review (randomised) and of rate increases
5. AI inbox triage: an intent classifier with a confidence threshold, a human-in-the-loop queue and an eval
"""
import json, os, sys
import numpy as np, pandas as pd, duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_union

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import data as D  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def bind_book(L, policy="broadcast"):
    s = D.submissions(L, policy=policy)
    s["commission_y1"] = np.where(s.bound, s.est_premium * s.line.map({k: v[1] for k, v in D.LINES.items()}), 0)
    return s


# ---------- 1. Metric tree & unit economics ----------
def unit_economics(L, S, renewal_rate):
    cpl = L.channel.map({k: v[0] for k, v in D.CHANNELS.items()})
    spend = L.assign(cpl=cpl).groupby("channel").cpl.sum()
    years = 1 / (1 - renewal_rate)                            # expected policy lifetime in years
    t = duckdb.sql("""
        select l.channel, count(*) leads, sum(l.contacted::int) contacted, sum(l.applied::int) applied,
               sum(coalesce(s.quotes,0) > 0) quoted, sum(coalesce(s.bound,false)::int) bound,
               sum(coalesce(s.commission_y1,0)) commission_y1
        from L l left join S s using(lead_id) group by 1 order by 1""").df().set_index("channel")
    t["spend"] = spend
    t["lead_to_bind_pct"] = 100 * t.bound / t.leads
    t["cac_per_policy"] = t.spend / t.bound
    t["commission_per_policy"] = t.commission_y1 / t.bound
    t["ltv_commission"] = t.commission_per_policy * years
    t["ltv_to_cac"] = t.ltv_commission / t.cac_per_policy
    t["payback_months"] = 12 * t.cac_per_policy / t.commission_per_policy
    funnel = {k: int(t[k].sum()) for k in ["leads", "contacted", "applied", "quoted", "bound"]}
    return t.round(2), funnel


# ---------- 2. Speed to lead ----------
def speed_to_lead(L):
    b = pd.cut(L.first_response_min, [0, 5, 60, 240, 1e9], labels=["<5 min", "5-60 min", "1-4 h", ">4 h"])
    S = bind_book(L)
    m = L[["lead_id"]].assign(bucket=b).merge(S[["lead_id", "bound"]], how="left").fillna({"bound": False})
    by = m.groupby("bucket", observed=True).bound.mean().mul(100).round(2)
    # Counterfactual: an AI agent answers after-hours leads within ~2 minutes (today they wait hours);
    # daytime leads stay with human SDRs. Both worlds are re-simulated with the SAME random draws (common random
    # numbers), so the only difference between them is response time.
    fast = np.random.default_rng(11).lognormal(np.log(2), .3, len(L))
    def world(ttr):
        q = L.quality.to_numpy()
        speed = np.select([ttr <= 5, ttr <= 60, ttr <= 24 * 60], [1.15, 1.0, .80], .60)
        r = np.random.default_rng(424242)      # independent stream (seed 0 would replay the lead generator)
        contacted = r.random(len(L)) < np.clip(.62 * speed * q ** .3, 0, .97)
        applied = contacted & (r.random(len(L)) < np.clip(.55 * q, 0, .95))
        return bind_book(L.assign(first_response_min=ttr, contacted=contacted, applied=applied))
    S = world(L.first_response_min.to_numpy())
    S2 = world(np.where(L.after_hours, fast, L.first_response_min))
    return by.to_dict(), dict(bound_now=int(S.bound.sum()), bound_ai_first_response=int(S2.bound.sum()),
                             commission_now=round(float(S.commission_y1.sum())),
                             commission_ai=round(float(S2.commission_y1.sum())),
                             after_hours_share=round(float(L.after_hours.mean()) * 100, 1),
                             after_hours_median_wait_h=round(float(L[L.after_hours].first_response_min.median() / 60), 1))


# ---------- 3. Carrier routing ----------
def routing(L):
    a, b = bind_book(L, "broadcast"), bind_book(L, "appetite")
    def summ(s):
        return dict(quote_rate=round(float((s.quotes > 0).mean()) * 100, 1),
                    avg_quotes=round(float(s.quotes.mean()), 2),
                    median_days_to_first_quote=round(float(s.first_quote_days.median()), 2),
                    bind_rate=round(float(s.bound.mean()) * 100, 1),
                    commission=round(float(s.commission_y1.sum())))
    by_ind = pd.DataFrame({"random": a.groupby("industry").bound.mean(), "appetite": b.groupby("industry").bound.mean()}).mul(100).round(1)
    return summ(a), summ(b), by_ind


# ---------- 4. Renewals ----------
def renewals(S):
    pol = D.renewals(S[S.bound].reset_index(drop=True))
    eff = pol.groupby("renewal_review").renewed.mean()
    n1, n0 = pol.renewal_review.sum(), (~pol.renewal_review).sum()
    se = np.sqrt(eff[True] * (1 - eff[True]) / n1 + eff[False] * (1 - eff[False]) / n0)
    band = pd.cut(pol.rate_change, [-1, .05, .10, .15, 1], labels=["<5%", "5-10%", "10-15%", ">15%"])
    by_rate = pol.groupby(band, observed=True).renewed.mean().mul(100).round(1)
    return pol, dict(policies=len(pol), renewal_rate=round(float(pol.renewed.mean()) * 100, 1),
                     review_vs_no_review=[round(eff[True] * 100, 1), round(eff[False] * 100, 1)],
                     review_lift_pts_95ci=[round((eff[True] - eff[False] - 1.96 * se) * 100, 1),
                                           round((eff[True] - eff[False] + 1.96 * se) * 100, 1)],
                     renewal_by_rate_change=by_rate.to_dict())


# ---------- 5. AI inbox triage ----------
KEYWORDS = {"claim_report": ["accident", "hurt", "injur", "slipped", "damage", "fire", "stole", "broke into", "suing", "sued", "rear-ended"], "cancel": ["cancel", "stop my policy"],
            "certificate_request": ["coi", "certificate", "proof of insurance"], "billing": ["charged", "payment", "premium", "monthly"],
            "policy_change": ["add", "update", "moved", "hired"], "new_quote": ["quote", "cost", "pricing", "looking for"]}


def rules(text):
    t = text.lower()
    for intent, kws in KEYWORDS.items():
        if any(k in t for k in kws):
            return intent
    return "new_quote"


def _fit(tr):
    vec = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                     TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True))
    clf = LogisticRegression(max_iter=3000, C=4).fit(vec.fit_transform(tr.text), tr.intent)
    return vec, clf


def _score(vec, clf, te, threshold):
    P = clf.predict_proba(vec.transform(te.text))
    te = te.assign(pred=clf.classes_[P.argmax(1)], conf=P.max(1), rule=te.text.map(rules))
    auto = (te.conf >= threshold) & (te.pred != "claim_report")   # claims always go to a human
    claims = te.intent == "claim_report"
    return te, dict(messages=len(te),
                    rules_accuracy=round((te.rule == te.intent).mean() * 100, 1),
                    model_accuracy=round((te.pred == te.intent).mean() * 100, 1),
                    automated_pct=round(auto.mean() * 100, 1),
                    auto_accuracy_pct=round((te.pred[auto] == te.intent[auto]).mean() * 100, 1) if auto.any() else None,
                    claim_recall_pct=round(((te.pred == "claim_report") & claims).sum() / max(claims.sum(), 1) * 100, 1),
                    # guardrail: a claim keyword list overrides the model and sends the message to a human
                    claim_recall_with_guardrail_pct=round((((te.pred == "claim_report") | (te.rule == "claim_report")) & claims).sum()
                                                          / max(claims.sum(), 1) * 100, 1))


def triage(msgs, threshold=0.80):
    """Two evals of the same agent.
    A) familiar wording: random 60/40 split, so test messages use phrasings seen in training
    B) new wording (drift): the last 2 phrasings of every intent never appear in training"""
    tr, te = train_test_split(msgs, test_size=.4, random_state=0, stratify=msgs.intent)
    vec, clf = _fit(tr)
    te_a, a = _score(vec, clf, te, threshold)
    sweep = []
    for th in [.5, .7, .8, .9, .95]:
        au = (te_a.conf >= th) & (te_a.pred != "claim_report")
        sweep.append(dict(threshold=th, automated_pct=round(au.mean() * 100, 1),
                          auto_accuracy_pct=round((te_a.pred[au] == te_a.intent[au]).mean() * 100, 1)))
    mixed = te_a.text.str.contains(" also ")
    a["accuracy_on_mixed_intent_msgs"] = round((te_a.pred[mixed] == te_a.intent[mixed]).mean() * 100, 1)
    n_t = msgs.groupby("intent").template_id.transform("max") + 1
    held = msgs.template_id >= n_t - 2
    vec2, clf2 = _fit(msgs[~held])
    _, b = _score(vec2, clf2, msgs[held], threshold)
    return dict(threshold=threshold, familiar_wording=a, new_wording=b, threshold_sweep_familiar=sweep)


def main():
    L = D.leads()
    S = bind_book(L)
    pol, ren = renewals(S)
    econ, funnel = unit_economics(L, S, ren["renewal_rate"] / 100)
    speed_by, speed = speed_to_lead(L)
    r_rand, r_app, by_ind = routing(L)
    tri = triage(D.inbox())
    econ.to_csv(f"{OUT}/unit_economics_by_channel.csv")
    by_ind.to_csv(f"{OUT}/bind_rate_by_industry_routing.csv")
    summary = dict(
        funnel=funnel,
        unit_economics={c: dict(cac=float(r.cac_per_policy), ltv=float(r.ltv_commission), ltv_to_cac=float(r.ltv_to_cac),
                                payback_months=float(r.payback_months), lead_to_bind_pct=float(r.lead_to_bind_pct))
                        for c, r in econ.iterrows()},
        speed_to_lead=dict(bind_pct_by_first_response=speed_by, **speed),
        carrier_routing=dict(random=r_rand, appetite=r_app),
        renewals=ren, ai_triage=tri)
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))
    return summary


if __name__ == "__main__":
    main()
