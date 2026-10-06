# Mixed BX/BR generation and proportional evaluation

Formal regional search now freezes the current regional parent pools once per
round, generates both BX and BR, and filters/selects their combined candidate
pool exactly once. Initialization uses archive_target_distinct slots, including
the fixed seed when enabled. Each I1 is generated, feature-checked and evaluated
before the next request, whose context contains only the most recent ten
successfully evaluated ideas. Invalid or duplicate slots are not refilled;
equal objectives and equal behaviors do not trigger initialization retries.
Immediately afterwards valid evaluated behavior points are partitioned, with
fewer regions if fewer distinct points are available. There are no BE rounds
or score-diversity/warmup gates. There is no separate global population or
pop_size parameter. Without regions, BX/BR sample from the current advantage
archive and each requests up to its actual size, using the same evaluation ratio.

For a region with \`n_r\` unique parents, the plan has at most \`n_r\` BX rows
and \`n_r\` BR rows. The existing probability based parent selection remains in
use. BX cycles the first parent by quality and samples additional parents from
the distance ranking with the existing seeded distribution. BR keeps the best
regional anchor and samples its second parent toward nearby behavior. A
singleton can produce one single-parent BX; BR keeps its existing two-parent
requirement and skips a singleton. Every round includes both intra-region and
cross-region BX. Each region splits its n_r BX slots equally; an odd extra slot
alternates between the two modes on successive rounds. With only one active
region all BX slots are intra-region. Unfilled parent-combination slots transfer
to the other mode without exceeding n_r. Both BX modes share duplicate keys.
Each operator uses its own no-repeat
parent signatures, and BX and BR are generated into one candidate pool.

Deduplication is per round and per operator, using an unordered canonical code
tuple, matching the existing parent-batch mechanism. Reversed pairs therefore
share the same duplicate key; no extra reversal pass is made.
The same pair may be used by BX and BR. Duplicate membership from fallback pools
is resolved by stable region order. Existing parent-pool code/score deduplication
and anchor inclusion remain unchanged; the archive target is not an exact count
of available parents (regional anchor references are maintained separately).
The advantage archive itself strictly retains at most archive_target_distinct
algorithms, one per observed objective, ordered by real performance. Ties do not
expand its capacity; fewer distinct scores leave the archive below capacity.
Failed generation is not refilled with extra parent pairs.
The existing global generated-algorithm safety limit may truncate a plan.

With region parent counts \`n_r\`, BX has an upper bound of
\`sum_r n_r\`. BR has an upper bound of
\`sum_{r:n_r>1}(n_r-1)\` when its first parent remains the regional best.
For 4+4+4 this is at most 12 BX and 9 BR, or 21 combined. For 5+5+6 it is
at most 16 BX and 13 BR, or 29 combined. If a valid probability-selected
parent set is unavailable, the corresponding slot is omitted.

`method.generation.evaluation_ratio=0.3` replaces the fixed formal evaluation
batch size. N counts parseable, unique, fresh candidates with successful behavior
extraction, BEFORE behavior filtering (so filtering does not reduce the budget
twice). B = min(ceil(ratio*N), remaining global evaluations, surviving candidates).
Empty pools get zero. Ratio must be finite and in (0,1]. If novelty is enabled,
B>=2 and candidates>B, reserve one of these B slots for the unchanged novelty
selector, excluding regional selections. The rest use the existing annealed
regional allocator and regional predictor ranking. No extra novelty evaluation
is added beyond B. The global successful-evaluation limit remains unchanged.

The default archive target is 24 (matching the current server configuration), regions 3, total successful
evaluation limit 210. Old candidates_per_region and eval_batch_size are removed
from the default regional configuration. The fixed operator cycle is no longer
called by formal search; legacy helper paths remain for this first revision.
The existing distance-rank parent-selection parameters remain in use.

Archive snapshots replace population management. Historical
population_generation_*.json names now contain the strict advantage archive;
best_population_generation_*.json is its incumbent. Full evaluated history,
predictor training, novelty and regional anchors retain their separate roles.

Logs: MixedParentPlan (per-region BX/BR counts), EvaluationRatio (N, ratio,
available, B), RegionalBatchBudget (novelty/region split). Lineage keeps actual
bx/br operators and pipeline.generation_kind. Tests mock LLM/real evaluator;
they do not establish optimization quality or real LLM validity rates.
