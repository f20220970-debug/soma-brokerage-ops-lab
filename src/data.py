"""Simulated US small-business insurance brokerage. Nothing here is Soma AI data.

Entities
- leads:        inbound small businesses by channel, state, industry, line of business, with time-to-first-response
- submissions:  each lead's application sent to carriers; carrier quote / decline and turnaround
- policies:     bound policies with annual premium and commission; renewal outcome a year later
- inbox:        inbound customer messages (email / SMS / call transcript snippets) with true intent labels,
                used to evaluate an AI triage agent
"""
import numpy as np, pandas as pd

LINES = {  # line of business: (median annual premium USD, commission rate, base bind rate)
    "General Liability":  (900,   .15, .34),
    "BOP":                (1600,  .15, .30),
    "Workers' Comp":      (3200,  .10, .26),
    "Commercial Auto":    (5200,  .10, .20),
    "Professional / E&O": (1800,  .15, .28),
}
CHANNELS = {  # cost per lead USD, lead quality multiplier, share of leads
    "Paid search":     (55, 1.00, .34),
    "Partner referral":(30, 1.35, .18),
    "Outbound":        (22, 0.70, .26),
    "Organic / SEO":   (8,  1.10, .14),
    "Marketplace":     (40, 0.85, .08),
}
INDUSTRIES = ["Contractors", "Restaurants", "Retail", "Professional services", "Trucking", "Health & wellness"]
CARRIERS = ["Carrier A", "Carrier B", "Carrier C", "Carrier D", "Carrier E", "Carrier F"]
# carrier appetite: probability of quoting per (industry) - the matching problem
APPETITE = np.array([
    # Contr Rest  Retail Prof  Truck Health
    [.75, .30, .60, .80, .10, .55],   # A
    [.40, .80, .55, .45, .05, .40],   # B
    [.65, .20, .35, .30, .70, .25],   # C
    [.30, .55, .75, .60, .15, .70],   # D
    [.55, .45, .40, .85, .35, .60],   # E
    [.20, .35, .30, .25, .80, .20],   # F
])
CARRIER_DAYS = np.array([2.0, 3.5, 4.0, 1.5, 5.0, 3.0])   # median quote turnaround, days
STATES = ["TX", "FL", "CA", "NY", "GA", "NC", "AZ", "OH", "IL", "NJ"]


def leads(seed=0, n=60000):
    r = np.random.default_rng(seed)
    ch = r.choice(list(CHANNELS), n, p=[v[2] for v in CHANNELS.values()])
    lob = r.choice(list(LINES), n, p=[.34, .22, .18, .12, .14])
    ind = r.choice(INDUSTRIES, n, p=[.24, .18, .17, .2, .09, .12])
    lob = np.where(ind == "Trucking", "Commercial Auto", lob)
    month = r.integers(0, 12, n)
    # time to first response (minutes): human SDR queue, long tail after hours
    after_hours = r.random(n) < .38
    ttr = np.where(after_hours, r.lognormal(5.6, .7, n), r.lognormal(3.0, 1.0, n))
    quality = np.array([CHANNELS[c][1] for c in ch])
    base = np.array([LINES[l][2] for l in lob])
    speed = np.select([ttr <= 5, ttr <= 60, ttr <= 24 * 60], [1.15, 1.0, .80], .60)
    p_contact = np.clip(.62 * speed * quality ** .3, 0, .97)
    contacted = r.random(n) < p_contact
    p_app = np.clip(.55 * quality, 0, .95)
    applied = contacted & (r.random(n) < p_app)
    premium = np.array([LINES[l][0] for l in lob]) * r.lognormal(0, .45, n)
    return pd.DataFrame(dict(lead_id=np.arange(n), month=month, channel=ch, line=lob, industry=ind,
                             state=r.choice(STATES, n), first_response_min=ttr.round(1), after_hours=after_hours,
                             contacted=contacted, applied=applied, est_premium=premium.round(0),
                             base_bind=base, quality=quality))


def submissions(L, policy="broadcast", seed=1, k=3):
    """Send each application to k carriers. policy='broadcast' picks carriers at random (no appetite model);
    policy='appetite' picks the k carriers with the best expected quote probability for the industry."""
    r = np.random.default_rng(seed)
    apps = L[L.applied].reset_index(drop=True)
    ind_idx = apps.industry.map({v: i for i, v in enumerate(INDUSTRIES)}).values
    rows = []
    for i, row in enumerate(apps.itertuples()):
        if policy == "appetite":
            score = APPETITE[:, ind_idx[i]] / np.sqrt(CARRIER_DAYS)
            chosen = np.argsort(-score)[:k]
        else:
            chosen = r.choice(len(CARRIERS), k, replace=False)
        quotes = []
        for c in chosen:
            q = r.random() < APPETITE[c, ind_idx[i]]
            days = r.lognormal(np.log(CARRIER_DAYS[c]), .4)
            quotes.append((c, q, days))
        quoted = [q for q in quotes if q[1]]
        n_quotes = len(quoted)
        first_quote_days = min(q[2] for q in quoted) if quoted else np.nan
        # bind: needs a quote; more quotes = better price; slower quotes lose the customer
        p_bind = 0 if not quoted else np.clip(
            row.base_bind * 2.4 * (1 + .18 * (n_quotes - 1)) * np.exp(-.12 * first_quote_days) * row.quality ** .5, 0, .95)
        bound = r.random() < p_bind
        rows.append((row.lead_id, n_quotes, first_quote_days, bound))
    s = pd.DataFrame(rows, columns=["lead_id", "quotes", "first_quote_days", "bound"])
    return apps.merge(s, on="lead_id")


def renewals(policies, seed=2):
    r = np.random.default_rng(seed)
    n = len(policies)
    claim = r.random(n) < .11
    touch = r.random(n) < .5                      # proactive renewal review 60 days out (randomised)
    rate_change = r.normal(.07, .06, n) + .12 * claim
    p = np.clip(.86 - 1.6 * np.maximum(rate_change - .05, 0) + .06 * touch - .05 * claim, .2, .98)
    renewed = r.random(n) < p
    return policies.assign(had_claim=claim, renewal_review=touch, rate_change=rate_change.round(3), renewed=renewed)


INTENTS = ["new_quote", "certificate_request", "policy_change", "claim_report", "billing", "cancel"]
_TEMPL = {
    "new_quote": ["need a quote for general liability for my {biz}", "how much would workers comp cost for {n} employees",
                  "looking for insurance for my new {biz}", "can you quote commercial auto for {n} trucks", "pricing for a BOP policy?",
                  "what's the rate for liability coverage", "opening a {biz} next month, what coverage do I need",
                  "shopping around for business insurance", "my landlord says I need insurance before I sign the lease"],
    "certificate_request": ["need a COI for a job site tomorrow", "please send a certificate of insurance to my client",
                            "general contractor is asking for proof of insurance", "can you add {co} as certificate holder",
                            "the venue wants our insurance paperwork before the event", "send the cert to {co} today please",
                            "client won't pay until they get our insurance certificate", "need the acord 25 form"],
    "policy_change": ["add a new vehicle to my policy", "we moved to a new address, update the policy",
                      "add an additional insured to the GL policy", "we hired {n} more employees, update workers comp",
                      "remove a driver from the auto policy", "we bought another van", "change the business name on the policy",
                      "our payroll went up, do I need to tell you"],
    "claim_report": ["a customer slipped in my {biz}, what do I do", "one of our trucks was in an accident",
                     "employee got hurt on the job today", "water damage at the shop last night, need to file",
                     "someone broke into the {biz} and stole equipment", "a client is suing us over the work we did",
                     "kitchen fire this morning", "our driver rear-ended someone"],
    "billing": ["my card was charged twice", "when is my next payment due", "can I switch to monthly payments",
                "why did my premium go up this month", "I need an invoice for my accountant", "update my card on file",
                "got a late payment notice", "is there a fee for paying in installments"],
    "cancel": ["I want to cancel my policy", "closing the business, please cancel coverage", "found cheaper, cancel at renewal",
               "stop my policy from renewing", "we sold the company, end the coverage", "don't need the auto policy anymore",
               "please terminate effective end of month", "going with another broker"],
}
_NOISE = ["hi", "hello team", "urgent", "thanks", "pls", "asap", "quick question:", "following up -", ""]


def inbox(seed=3, n=6000):
    r = np.random.default_rng(seed)
    p = np.array([.24, .26, .18, .07, .17, .08])
    rows = []
    for _ in range(n):
        it = r.choice(INTENTS, p=p)
        k = int(r.integers(len(_TEMPL[it])))
        t = _TEMPL[it][k].format(biz=r.choice(["restaurant", "salon", "store", "gym", "office", "shop"]),
                                        n=r.integers(2, 40), co=r.choice(["ABC Builders", "Lone Star GC", "Metro Dev"]))
        # some messages mix intents (e.g. a cancel phrased as a price complaint) to make triage hard
        if r.random() < .12:
            other = r.choice([x for x in INTENTS if x != it])
            t = t + " also " + r.choice(_TEMPL[other]).format(biz="store", n=5, co="ABC Builders")
        msg = f"{r.choice(_NOISE)} {t}".strip()
        if r.random() < .3:
            msg = msg.lower()
        rows.append((msg, it, k, r.choice(["email", "sms", "call"], p=[.5, .3, .2])))
    return pd.DataFrame(rows, columns=["text", "intent", "template_id", "channel"])
