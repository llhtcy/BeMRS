# Mixed BX/BR generation and proportional evaluation

Formal regional search now freezes the current regional parent pools once per
round, generates both BX and BR, and filters/selects their combined candidate
pool exactly once. The initialization and the two behavior-exploration rounds
retain their existing settings. Without initialized regions the original global
parent fallback remains, but formal evaluation uses the ratio there as well.

For a region with \`n_r\` unique parents, the plan has at most \`n_r\` BX rows
and \`n_r\` BR rows. Each member is the first parent of one BX row, paired with
its farthest regional neighbor. BR keeps the best regional anchor and pairs it
with each remaining parent; its anchor is the single-parent row when the region
is a singleton. Singletons therefore have one single-parent BX and one
single-parent BR. No cross-region rows are added to this per-region plan.

Deduplication is per round and per operator, with ordered code tuples. Reversed
pairs remain distinct when they arise naturally; no extra reversal pass is made.
The same pair may be used by BX and BR. Duplicate membership from fallback pools
is resolved by stable region order. Existing parent-pool code/score deduplication
and anchor inclusion remain unchanged; the archive target is not an exact count
of available parents. Failed generation is not refilled with extra parent pairs.
The existing global generated-algorithm safety limit may truncate a plan.

With region parent counts \`n_r\`, planned offspring count is
\`2 * sum_r n_r\`. For 4+4+4 this is 24; for 5+5+6 it is 32. If a valid
two-parent combination is unavailable, the corresponding slot is omitted.

`method.generation.evaluation_ratio=0.2` replaces the fixed formal evaluation
batch size. N counts parseable, unique, fresh candidates with successful behavior
extraction, BEFORE behavior filtering (so filtering does not reduce the budget
twice). B = min(ceil(ratio*N), remaining global evaluations, surviving candidates).
Empty pools get zero. Ratio must be finite and in (0,1]. If novelty is enabled,
B>=2 and candidates>B, reserve one of these B slots for the unchanged novelty
selector, excluding regional selections. The rest use the existing annealed
regional allocator and regional predictor ranking. No extra novelty evaluation
is added beyond B. The global successful-evaluation limit remains unchanged.

The default archive target is 12, population 10, regions 3, total successful
evaluation limit 210. Old candidates_per_region and eval_batch_size are removed
from the default regional configuration. The fixed operator cycle is no longer
called by formal search; legacy helper paths remain for this first revision.
Fallback distance-rank parameters do not participate in the new regional plan.

Logs: MixedParentPlan (per-region BX/BR counts), EvaluationRatio (N, ratio,
available, B), RegionalBatchBudget (novelty/region split). Lineage keeps actual
bx/br operators and pipeline.generation_kind. Tests mock LLM/real evaluator;
they do not establish optimization quality or real LLM validity rates.
