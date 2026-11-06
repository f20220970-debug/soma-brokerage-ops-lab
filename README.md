# Soma Brokerage Ops Lab

[![tests](https://github.com/f20220970-debug/soma-brokerage-ops-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/f20220970-debug/soma-brokerage-ops-lab/actions/workflows/tests.yml)

Built for **Soma AI's Founder's Office (Generalist) internship**. It works through five problems a fast-scaling, AI-native business-insurance brokerage faces, across growth, sales, operations and AI agents. Each one ends in a decision and the metric to track.

> **All data is simulated**: 60,000 inbound US small-business leads, 6 carriers, 5 lines of business and 6,000 inbox messages, with made-up behaviour. Nothing here is Soma AI data. The numbers show method and the kind of call each analysis supports, not facts about Soma.

Run it: `pip install -r requirements.txt && python run.py && pytest -q` (about 10 seconds). Outputs go to `results/`.

## 1. Metric tree and unit economics by channel

Funnel: **60,000 leads → 33,761 contacted → 18,395 applied → 15,854 quoted → 9,067 bound.**

| Channel | Lead → bind | CAC per bound policy | Commission LTV | LTV / CAC | Payback |
| --- | --- | --- | --- | --- | --- |
| Partner referral | 25.1% | $119 | $1,461 | **12.2×** | 5 months |
| Organic / SEO | 18.0% | $45 | $1,451 | **32.6×** | 2 months |
| Paid search | 15.2% | $362 | $1,461 | 4.0× | 16 months |
| Outbound | 7.9% | $278 | $1,446 | 5.2× | 12 months |
| Marketplace | 10.3% | $388 | $1,471 | 3.8× | 17 months |

LTV is first-year commission × expected policy life (1 ÷ (1 − 80.9% renewal)).

**Call:** every channel pays back, but referral and SEO pay back in under 6 months. Scale partnerships and content first, and keep paid search to the industries where it binds best.

## 2. Speed to lead: what an always-on AI first response is worth

- **Today:** 37.8% of leads arrive after hours and wait a **median of 4.5 hours** for a human. Leads answered in under 5 minutes bind at **18.5%**, against **12.7%** for those answered after more than 4 hours.
- **Counterfactual:** an AI agent answers after-hours leads in about 2 minutes, while daytime stays with human SDRs. Both cases use the same random draws, so response time is the only difference.
- **Result:** bound policies rise from **9,204 to 10,465 (+13.7%)** and first-year commission rises **+14.9%**.

**Call:** the cheapest growth lever here is coverage, not more leads.

## 3. Carrier routing by appetite

Each application goes to 3 carriers. Today they're picked without an appetite model; the alternative ranks carriers by their likelihood of quoting that industry, adjusted for turnaround.

| | Random | **By appetite** |
| --- | --- | --- |
| Applications with ≥1 quote | 86.2% | **95.9%** |
| Quotes per application | 1.41 | **1.92** |
| Median days to first quote | 2.29 | **1.83** |
| Bind rate | 49.3% | **60.7%** |
| First-year commission | $2.52M | **$3.15M (+25%)** |

Trucking gains most (27.5% → 38.7% bind), because only 2 of the 6 carriers want it.

## 4. Renewals

- **Proactive review:** a review 60 days before renewal (randomised 50/50) lifts renewal from **77.5% to 84.3%** (+6.8 pts, 95% CI 5.2–8.4).
- **Rate increases:** renewal falls from **87.9%** (under 5% increase) to **62.1%** (over 15%).

**Call:** prioritise reviews by size of rate increase, and re-market policies facing increases above 10%.

## 5. AI inbox triage, and why the eval matters more than the model

An intent classifier (word and character TF-IDF with logistic regression) sorts customer messages into 6 intents: quote, certificate, policy change, claim, billing and cancel.

| | Familiar wording (random split) | **New wording** (2 phrasings per intent never seen in training) |
| --- | --- | --- |
| Model accuracy | 98.5% | **29.1%** |
| Keyword rules | 67.9% | 47.8% |
| Auto-handled at confidence ≥ 0.8 | 87.8%, at 99.9% accuracy | 4.9%, at **0% accuracy** |
| Claims caught (model only) | 92.9% | 44.6% |
| Claims caught (model + claim keyword guardrail) | **100%** | **100%** |

**What this shows:**
- **A random-split eval overstates readiness.** On new wording, the model is wrong and *confident*, so a confidence threshold alone isn't a safety net.
- **The design:** auto-handle only high-confidence, low-risk intents. Send every possible claim to a human via a keyword guardrail. Hand-label a sample of 100 real messages every week to catch drift before customers do.

## Files

```
run.py                     all five analyses -> results/
src/data.py                the simulator (assumptions in one place)
docs/founders_office_playbook.md   weekly metric review, 30-60-90 plan, experiment backlog
tests/test_lab.py          5 checks (funnel monotonic, routing beats random, guardrail catches all claims, ...)
```

Companion repos: [revenue-pipeline-forecaster](https://github.com/f20220970-debug/revenue-pipeline-forecaster) and [arr-growth-model](https://github.com/f20220970-debug/arr-growth-model).
