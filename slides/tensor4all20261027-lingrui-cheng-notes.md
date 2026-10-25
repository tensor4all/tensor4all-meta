# tensor4all-rs: tree methods and adaptive patching

Lingrui Cheng

tensor4all developer meeting, 27 October 2026

## 1. tensor4all-rs: tree methods and adaptive patching

[About 15 seconds]
This report presents Lingrui Cheng's contributions to tensor4all-rs since the September 22 meeting, together with subsequent integration by other tensor4all developers. The meeting date is October 27, 2026. The report covers library development: tree adaptive patching, TreeRSI, numerical reliability, and the remaining implementation and validation work.

Sources:

- https://github.com/tensor4all/tensor4all-rs

## 2. Since September 22

[About 30 seconds]
This update continues from the September 22 meeting. It focuses on tree algorithms in the Rust library, developed by Lingrui Cheng in collaboration with Hiroshi Shinaoka and the tensor4all developers. ACI and SRC were already present at the previous meeting. The new work is adaptive patched interpolation on trees, the new TreeRSI implementation, and fixes that make the existing methods more usable. The original ACI, SRC, RSI and patching papers formulate their algorithms for tensor trains. Cheng's contribution is the generalization to branching tree tensor networks and its implementation. Subsequent integration by other developers is distinguished from the initial contribution.

Sources:

- https://github.com/tensor4all/tensor4all-rs/pull/900
- https://github.com/tensor4all/tensor4all-rs/pull/907
- https://github.com/tensor4all/tensor4all-rs/pull/646
- https://github.com/tensor4all/tensor4all-rs/pull/694
- https://arxiv.org/abs/2604.00037
- https://arxiv.org/abs/2504.06475

## 3. Adaptive patching on trees

[About 1 minute 30 seconds]
The main addition is adaptive patched interpolation on arbitrary trees. The patch driver asks an engine to interpolate one region. It accepts the result or fixes another site and creates children. Fixed sites do not require deleting tree vertices, so junctions remain connected and all patches keep the same named topology. The engine interface lives in treetn and the driver has no dependency on TreeTCI itself.
The driver owns the function-value cache. A split transfers the samples into the appropriate children, rather than reevaluating the same points. It can also recycle pivots and use the largest inherited cached values as starting candidates.
For L2 acceptance, all patches share a pinned reference and the driver measures the residual itself. Small regions can use exhaustive checks. Large regions use samples and an independent audit, which yield estimates rather than a general guarantee. A localized residual can escape both. The patch-size controls prevent indefinite subdivision: a retained patch that fails its allowance is explicitly reported as ToleranceNotMet.
Cheng's interpolation-line contribution merged in PR 900. The compact representation decision remains deferred. The patched-algebra global error contract is also separate work.

Sources:

- https://github.com/tensor4all/tensor4all-rs/pull/900
- https://github.com/tensor4all/tensor4all-rs/blob/main/docs/design/tree-adaptive-patching-roadmap.md
- https://github.com/tensor4all/tensor4all-rs/blob/main/docs/design/tree-patching-error-contract.md
- https://github.com/tensor4all/tensor4all-rs/blob/main/docs/design/tree-pqtci-patch-size-bounds.md
- https://arxiv.org/abs/2602.22372

## 4. Patching integration and execution

[About 1 minute]
Cheng contributed the sequential tree interpolation driver. Shinaoka subsequently integrated local and MPI execution through hataori. The old TT-only partitionedtt crate has been removed, so chains and branched trees now share one implementation. The patch driver depends on a common interpolation-engine interface rather than on TreeTCI itself.
Parallel execution distributes independent patches while keeping a consistent decision order. The numerical work follows the caller's execution resources. Tests compare serial and local execution, and MPI smoke tests cover one and two ranks. These establish functional integration, not parallel speedup. Broader reproducibility and scaling measurements remain open.

Sources:

- https://github.com/tensor4all/tensor4all-rs/pull/900
- https://github.com/tensor4all/tensor4all-rs/pull/1005
- https://github.com/tensor4all/tensor4all-rs/pull/1009
- https://github.com/tensor4all/tensor4all-rs/blob/main/crates/tensor4all-partitionedtreetn/README.md

## 5. TreeRSI

[About 1 minute 15 seconds]
TreeRSI is the other new algorithm in this period. It applies recursive sketched interpolation to simultaneous elementwise products of compatible TreeTNs. The original method is a tensor-train algorithm. On a branching tree, a message must account for the other branches: the implementation handles child and sibling information and the downward environment. The leaf-root tests matter because a junction then has both a parent and siblings, which exercises a different recurrence from a junction-root example.
To separate implementation differences from random probes, the validation records the NumPy draws in the authors' code and replays them in Rust. The comparison includes 56 DMRG case/cap points and 22 synthetic points. Agreement is at numerical precision within the numerical rank. Three synthetic cases differ after numerical rank, where zero-tolerance LU selects noise-level pivots.
The branched checks compare against the exact represented product and against TreeACI when both results are accurate. The new two-revision benchmark adds timing evidence: RSI is 2.89 to 4.83 times faster than current TreeACI on four synthetic f64 products, using two operands on chains and binary trees. All four caps are at or above the recorded product cut rank, and the independent dense errors are near machine precision. This supports a claim about these full-reconstruction fixtures, not a speed advantage for truncating compression. Earlier tests still show lower fixed-cap accuracy and wider seed spread on some branched problems.

Sources:

- https://github.com/tensor4all/tensor4all-rs/pull/907
- https://github.com/tensor4all/tensor4all-rs/pull/1002
- https://github.com/tensor4all/tensor4all-rs/pull/1012
- https://github.com/tensor4all/tensor4all-rs/blob/main/docs/design/tree-rsi.md
- https://arxiv.org/abs/2602.17974
- t4a-bench-2026-10-27-REPORT.md, G4; fixture.json and observations.jsonl

## 6. Reliability of the tree methods

[About 1 minute 15 seconds]
The repairs fall into three practical groups. First, stronger searches and stopping checks reduce false convergence: TreeTCI keeps improvements while searching and must use newly found pivots before it stops. TreeACI rechecks earlier failing guard points instead of forgetting them after a later random miss. Results distinguish convergence from a rank or iteration limit.
Second, site-free nodes are important for genuine tree layouts and fixed-site patches. Cheng repaired junction rank growth and basic norms and decompositions. Shinaoka completed further operations, including inner products and SRC. This support is operation-specific; it does not mean every solver accepts site-free states.
Third, numerical fixes preserve small nonzero contributions, reject non-finite values and align tolerance scaling. Some TreeACI convergence issues remain open. The expensive previous-local-cross retention was withdrawn, while the independent guard-failure checks remain.

Sources:

- https://github.com/tensor4all/tensor4all-rs/pull/790
- https://github.com/tensor4all/tensor4all-rs/pull/839
- https://github.com/tensor4all/tensor4all-rs/pull/808
- https://github.com/tensor4all/tensor4all-rs/pull/799
- https://github.com/tensor4all/tensor4all-rs/pull/986
- https://github.com/tensor4all/tensor4all-rs/pull/838
- https://github.com/tensor4all/tensor4all-rs/pull/840
- https://github.com/tensor4all/tensor4all-rs/pull/855
- https://github.com/tensor4all/tensor4all-rs/pull/861
- https://github.com/tensor4all/tensor4all-rs/pull/895
- https://github.com/tensor4all/tensor4all-rs/issues/854

## 7. Evaluation and memory

[About 1 minute]
There are two distinct ways to avoid repeated work. The cached evaluator speeds up evaluating the current tensor-network approximation by reusing contractions. The optional target memo stores values returned by the original function, so repeated requests need not call that function again. The target memo is disabled by default and requires a stable value for each point during a run.
The costly-oracle benchmark illustrates the distinction. On a synthetic branch case, current main is 4.2 times faster than the old version with memo off, and 12.8 times faster with memo on. The additional memo benefit within current main is therefore about three times. This is evidence for an expensive synthetic callback, not a general throughput claim.
Finite cache and buffer policies reduce memory pressure. They also have workload-dependent overhead, shown on the later costs page. They do not impose an upper bound on total process memory.

Sources:

- https://github.com/tensor4all/tensor4all-rs/pull/798
- https://github.com/tensor4all/tensor4all-rs/pull/865
- https://github.com/tensor4all/tensor4all-rs/pull/852
- https://github.com/tensor4all/tensor4all-rs/pull/846
- t4a-bench-2026-10-27-REPORT.md, G2 and G5x

## 8. Benchmark: improvements

[About 1 minute 10 seconds]
These are selected improvements from the frozen two-revision comparison. The old revision is 4fee1a327ed6b1861fa1d0ea36a00559d0fe078f, main at the September 22 meeting, and the new revision is 9222cbeec2e62b512b546685a4b513b139757136, dated October 22. The two versions span 156 first-parent commits by the whole team, plus tenferro and hataori dependency updates and a change in CPU kernel provider. The bars are descriptive workload results and are not attributed entirely to one author.
The TreeTCI tree case drops from 0.8951 to 0.0540 seconds with similar independent error, a 16.6-times gain. The synthetic costly-oracle case drops from 725.2 to 56.8 milliseconds with target memo enabled, a 12.8-times gain. Current main with memo off is already 4.2 times faster than old. Cached readout and fewer target calls are distinct mechanisms. The separate 129-site chain in the report improves 14.9 times.
Warm hinted readout is a 64-point, fully cached chain query at chi=256 with identical explicit cache and batch settings: 401.945 to 87.885 microseconds. It benefits from the borrowed two-column cut path. Tree SRC is fixed-rank binary-tree MPO-MPO, 8.006 to 2.754 milliseconds, with equal measured error 6.606e-8. Profiled contraction signatures and call counts match while per-call einsum cost decreases; no single-PR attribution is established. Patched addition uses the same tree crate and exact-gain harness in both versions: 38.382 to 23.597 milliseconds, with unchanged patch counts and approximately 2e-15 error. Its mechanism is not isolated.
Every displayed bar uses the median of five alternating rounds on one pinned CPU thread. The original report retains ranges and cases with no gain or regressions. These results do not measure parallel scaling.

Sources:

- t4a-bench-2026-10-27-REPORT.md, G1/G2/G5x/G6/G7a2
- t4a-bench-2026-10-27/figures/speedup_rows.csv
- t4a-bench-2026-10-27/PROTOCOL.md
- https://github.com/tensor4all/tensor4all-rs/pull/798
- https://github.com/tensor4all/tensor4all-rs/pull/852
- https://github.com/tensor4all/tensor4all-rs/pull/865
- https://github.com/tensor4all/tensor4all-rs/commit/4fee1a327ed6b1861fa1d0ea36a00559d0fe078f
- https://github.com/tensor4all/tensor4all-rs/commit/9222cbeec2e62b512b546685a4b513b139757136

## 9. Benchmark: costs and limits

[About 1 minute 35 seconds]
The positive bars do not describe every workload. In the four high-rank Hadamard fixtures, current TreeACI is 1.50 to 2.08 times slower than the old version. Source inspection and the separate diagnostics runs identify the main reason: the old schedule skipped global Guard when no cut had injection capacity, whereas the new schedule still detects a residual and can report a rank limit. The cases found no additional pivots, so they show the cost of the added correctness check. Small branch-cost fixtures show larger slowdowns. For degree three, the median new/old time ratio over eight configurations is approximately 1.62, rather than the 1.15 ratio of the single least-regressed case selected in the supplied overview figure.
Default cache limits also have a cost. The new evaluator bounds message and slice caches and chunks work based on local tensor size. In the degree-three chi=64 fixture, the defaults yield 1.3 to 1.5 times longer execution, while the same explicit cache/chunk settings return performance to approximately parity. The limits protect memory but their effect depends on the topology and rank.
The right side is one patching counterexample, not a general verdict on adaptive patching. For eta=0.03 the old monolithic run takes 27.8525 seconds and the new patched run 111.0135 seconds. Independent 4,000-point relative L2 estimates are 3.1265e-5 and 3.8008e-4. Thus the patched run is both slower and less accurate here. Its oracle point count falls from 81.04 million to 13.91 million, but stored parameters grow from 2.015 million to 10.038 million. It also imposes cap 64, versus no cap on the monolithic arm, and uses a sampled-max driver criterion. This is a measured configuration tradeoff, not a matched-error speedup. A benefit for sufficiently expensive oracles remains a hypothesis for this workload.

Sources:

- t4a-bench-2026-10-27-REPORT.md, G3/G4/G5/G5x/G7b
- t4a-bench-2026-10-27/results/main/raw.jsonl
- t4a-bench-2026-10-27/diag/aci-diag-binary128-new.jsonl
- https://github.com/tensor4all/tensor4all-rs/pull/855
- https://github.com/tensor4all/tensor4all-rs/pull/846

## 10. Remaining work

[About 1 minute]
The interpolation line is in main: the engine seam, tree driver, measured-error reporting and patch-size controls. The compact patch representation is still a deferred decision. Later integration also provides local and MPI interpolation, so parallel execution is no longer entirely future work.
The largest algorithmic gap is the complete adaptive patched contraction workflow, especially patched MPO-MPO: exposing saturation early, refining only the inputs that caused a failed output, recomputing only those outputs and merging compatible neighbors within the error allowance. Existing contract_adaptive is useful groundwork but does not complete that roadmap.
The benchmark adds concrete performance questions: reduce Guard overhead while preserving its detection contract, study bounded-cache defaults on branching nodes, and understand patch overhead and storage on workloads with delocalized features. Matched-accuracy RSI timing in a genuinely truncating regime and multi-thread or MPI scaling remain open. The sampled-error blind spot and the remaining TreeACI convergence problems are also still priorities.

Sources:

- https://github.com/tensor4all/tensor4all-rs/blob/main/docs/design/tree-adaptive-patching-roadmap.md
- https://github.com/tensor4all/tensor4all-rs/blob/main/docs/design/treetn-contraction-outcome.md
- https://github.com/tensor4all/tensor4all-rs/blob/main/docs/design/tree-patching-error-contract.md
- https://github.com/tensor4all/tensor4all-rs/issues/854
- https://github.com/tensor4all/tensor4all-rs/pull/1005
- t4a-bench-2026-10-27-REPORT.md
