# Founder's Office playbook (built on this repo's simulated findings)

## 1. Weekly metric review: one page, every Monday

| Layer | Metric | Why it is on the page | Owner |
| --- | --- | --- | --- |
| North star | New annualised commission bound this week | What the business actually earns | Founder's Office |
| Growth | Leads by channel; CAC per bound policy; LTV/CAC | Where the next dollar should go | Growth |
| Sales | % of leads answered in < 5 min (day and night); lead → bind % | Speed is the biggest lever found | Sales |
| Ops | Applications with ≥ 1 quote; median days to first quote; quotes per application | Carrier routing quality | Ops |
| Retention | Renewal rate (rolling 90 days); renewals with rate change > 10% not yet reviewed | Revenue that is already won and can be lost | Account management |
| AI agents | % of messages auto-handled; accuracy on the weekly 100-message labelled sample; claims sent to a human within 5 minutes | Automation without hidden errors | Engineering + Ops |

**Rules for the review:**
- Every red metric gets one owner and one next action.
- Nothing is discussed without the trend for the last 8 weeks.

## 2. First 30 / 60 / 90 days in the role

- **Days 1–30: learn and measure.**
  - Shadow 20 sales calls and 10 service calls.
  - Rebuild the funnel and metric tree from production data, and reconcile it with finance.
  - Find the 3 biggest leaks.
- **Days 31–60: own one lever end to end.** For example, after-hours coverage by the AI agent:
  - Ship it to one state first.
  - Measure with a holdout.
  - Write the playbook.
- **Days 61–90: scale what worked.**
  - Roll out to all states.
  - Hand the metric to an owner.
  - Start the next lever, such as appetite-based carrier routing.

## 3. Experiment backlog (ranked by expected value ÷ effort)

| # | Experiment | Metric | Design | Expected effect (simulated) |
| --- | --- | --- | --- | --- |
| 1 | AI agent answers after-hours leads in under 2 min | Lead → bind | Alternate weeks per state (switchback) | +13.7% bound |
| 2 | Route submissions by carrier appetite | Bind rate, days to quote | 50/50 by application | +11 pts bind |
| 3 | Renewal review for rate increases > 10% | Renewal rate | Randomised by policy | +6.8 pts |
| 4 | Shift paid-search budget to partner programmes | CAC per policy | Geo split by state | CAC $362 → $119 per policy on shifted spend, if partner supply scales |
| 5 | Auto-handle certificate requests above the 0.9 confidence threshold | Time to send a certificate, error rate | Shadow mode for 2 weeks, then live | 85% automated in familiar-wording tests |

## 4. Things to verify with real data before believing any of this

- Does response time cause higher bind rates, or do good leads just get answered faster? Only a randomised test settles it.
- Carrier appetite changes. Refresh the model monthly from actual quote/decline outcomes.
- LTV assumes renewal rates hold. Check them by cohort, line of business and state.
