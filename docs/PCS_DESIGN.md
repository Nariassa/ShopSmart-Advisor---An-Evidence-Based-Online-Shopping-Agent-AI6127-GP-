# Purchase Confidence Score (PCS): Grounded Generation & Purchase Confidence Scoring design

PCS is an experimental evidence confidence score, not a calibrated probability that a purchase will be satisfactory. The proposal names evidence quality, consistency and recency but does not specify weights. The following formula is a tunable implementation choice.

## Formula

~~~text
PCS = 100 × V × C × (0.40 Q + 0.35 S + 0.25 R)
V   = n / (n + 5)
S   = 1 - P
P   = Σ_a w_a × 4 × p_positive,a × p_negative,a
C   = Σ_a w_a × min(n_a / 2, 1)
~~~

Aspect weights are non-negative and normalized over requested aspects. Defaults are equal weights. n is the deduplicated review evidence unit count in supplied evidence; specifications do not count. n_a is the unit count for aspect a. Positive, negative and neutral proportions use the total aspect count, including neutral opinions. An even positive/negative split gives P=1; unanimous polarity gives P=0. Missing requested aspects have zero coverage. No review evidence gives PCS=0 even if Aspect Analysis & Purchase Decision Explanation supplies favorable statistics.

V prevents a single consistent review from obtaining high confidence. True source_review_id is preferred for deduplication. Without original IDs, normalized text hashes are proxies only: distinct aspect fragments from one review can still be correlated. Outputs disclose missing IDs and call them evidence units, not independent users.

## Quality and recency

~~~text
q_i = 0.50 + 0.10 × min(words_i / 100, 1)
      + 0.20 × verified_i
      + 0.20 × min(log(1 + helpful_votes_i) / log(101), 1)
Q   = mean(q_i)
r_i = 0.5 ^ (age_days_i / 180)
R   = mean(r_i)
~~~

Unknown purchase verification and helpful votes receive no bonus. Missing, invalid or future review timestamps use r_i=0.5 and are counted in missing_metadata. Time is UTC; Unix seconds, milliseconds and ISO timestamps are supported. Product release dates are not review dates.

The delivered chunks.parquet has 4,450,749 rows and columns chunk_id, parent_asin, chunk_type, aspect, text, rating, category. It lacks original review IDs, timestamps, verified purchase and helpful votes. In the inspected product, rating is identical on specifications and reviews and may be the product average. Rating is therefore never used as aspect sentiment.

## Aspect-analysis integration

Per-chunk sentiment_label and signed sentiment_score override Grounded Generation & Purchase Confidence Scoring's sentiment baseline. Per-product aspects count distributions replace corresponding local distributions. Optional coverage and polarization scalars override coverage and polarization after validation in [0,1]. Q, R and V still use supplied review evidence, so full-corpus aspect-analysis aggregates plus retrieved-sample generation evidence must be reported with both scopes.

Neither polarization nor coverage is a positive product opinion. A reliably bad product can have high PCS. We separately return:

~~~text
fit_score      = Σ_a w_a × (p_positive,a + 0.5 × p_neutral,a)
decision_score = PCS × (0.25 + 0.75 × fit_score)
~~~

Missing aspect opinions use fit=0.5. Candidate ordering uses decision_score and puts confirmed budget violations last. Verdicts check evidence sufficiency, unfavorable opinions, unknown budget and unvalidated baseline sentiment. Current PCS=35 and fit=0.4/0.6 thresholds are heuristics, not calibrated decision boundaries.

## Limits and evaluation plan

Retrieved evidence is selected, not a representative population sample. Neutral or unclassified text can hide disagreement; the lexicon does not estimate sentiment uncertainty reliably. The system discloses baseline use and withholds unqualified positive advice from that baseline. Manufacturer statements remain attributed specifications.

System Integration, User Interface & Evaluation should use held-out annotations to compare PCS with judged evidence adequacy, not equate it with star ratings. Test weight sensitivity, volume prior, minimum coverage and time half-life. Ablate Q, S/P, R, V and C separately while holding evidence fixed. Evaluate sentiment and NLI independently, including negation, contradictions, mixed aspects and translated queries. Do not use unit tests or synthetic examples as benchmark accuracy estimates.
