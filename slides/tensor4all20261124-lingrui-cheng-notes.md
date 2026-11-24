# tensor4all-rs: patched algebra on trees

Lingrui Cheng

tensor4all meeting, 24 November 2026, 17:00 JST

Speaker notes for slides/tensor4all20261124-lingrui-cheng.pdf. Scope: pull requests and issues opened by Lingrui Cheng (lingrui96) in tensor4all-rs since the 27 October 2026 meeting.

## 1. tensor4all-rs: patched algebra on trees

[About 30 seconds]
This update covers my own work in tensor4all-rs since the October 27 meeting. At that meeting the biggest gap was patched algebra: multiplying and transforming patched networks end to end. Most of this period went into closing that gap, and into making TreeTCI and TreeACI report convergence honestly. I will also show the measurements that did not go our way.

Sources:

- https://github.com/tensor4all/tensor4all-rs

## 2. Activity since October 27

[About 30 seconds]
In numbers: 54 merged pull requests out of 59 opened, two still in review, and 39 issues, most of them already closed. The work touched 13 crates, but most of it sits in three places: partitionedtreetn for patched algebra, quanticstransform for the patched Fourier transform, and the TreeTCI and TreeACI solvers, where the theme was making convergence reports honest.

On the slide: Pull requests and issues opened by lingrui96 in tensor4all/tensor4all-rs, 27 October 2026 14:00 CET to 24 November 2026. Crates: those touched by the merged pull requests.

Sources:

- https://github.com/tensor4all/tensor4all-rs/pulls?q=author%3Alingrui96+created%3A%3E%3D2026-10-27

## 3. Status at a glance

[About 1 minute]
Before the details, the status of everything in this talk. Three things are implemented but not yet efficient: patched contraction, patched element-wise operations and the patched Fourier transform. They are in main and pass their accuracy checks, but on the workloads we measured they are slower than the unpatched route; slide 11 has the numbers. Sibling merging with per-patch bounds, the sum and lookup utilities, the patched QTT inputs and the convergence fixes are implemented. So are rSVD and rQR in core, which may move to tenferro; that is not decided yet. In progress: the tolerance merge policy in review and the TreeTCI sampled-stop threshold as a draft. Planned: per-region bounds for merge_siblings and two junction cases in TreeTCI and TreeACI. On hold: MPI patch parallelism for interpolative QTT; its design was approved, but the work is paused.

On the slide: “To optimize”: merged and correct on its checks, but slower than the unpatched route on the measured workloads (slide 11).

Sources:

- https://github.com/tensor4all/tensor4all-rs/pull/1278
- https://github.com/tensor4all/tensor4all-rs/pull/1277
- https://github.com/tensor4all/tensor4all-rs/pull/1287
- https://github.com/tensor4all/tensor4all-rs/pull/1296
- https://github.com/tensor4all/tensor4all-rs/pull/1330
- https://github.com/tensor4all/tensor4all-rs/pull/1369
- https://github.com/tensor4all/tensor4all-rs/pull/1113
- https://github.com/tensor4all/tensor4all-rs/pull/1114
- https://github.com/tensor4all/tensor4all-rs/pull/1115
- https://github.com/tensor4all/tensor4all-rs/pull/1117
- https://github.com/tensor4all/tensor4all-rs/pull/1116
- https://github.com/tensor4all/tensor4all-rs/pull/1302
- https://github.com/tensor4all/tensor4all-rs/pull/1307
- https://github.com/tensor4all/tensor4all-rs/pull/1357
- https://github.com/tensor4all/tensor4all-rs/pull/1434
- https://github.com/tensor4all/tensor4all-rs/pull/1440
- https://github.com/tensor4all/tensor4all-rs/pull/1344
- https://github.com/tensor4all/tensor4all-rs/pull/1407
- https://github.com/tensor4all/tensor4all-rs/pull/1439
- https://github.com/tensor4all/tensor4all-rs/issues/1443
- https://github.com/tensor4all/tensor4all-rs/issues/1435
- https://github.com/tensor4all/tensor4all-rs/issues/1436
- https://github.com/tensor4all/tensor4all-rs/issues/1342

## 4. Patched algebra on trees

[About 45 seconds]
Patched algebra in three parts, all merged in PR 1278, so all three are implemented. M6a: contraction returns an outcome, either a completed contraction with its edge ranks or the first edge that hit the rank threshold, so contract_adaptive can stop a group early instead of contracting every pair first. Overlapping output projectors are refined onto a disjoint grid instead of failing. M6b: sibling patches merge back while the L2 contract holds. M6c: element-wise functions of patched inputs with TreeACI or TreeRSI as the kernel. On every exhaustive check up to R = 7 the L2 contract holds. Contraction and element-wise are marked “to optimize”: they are correct, but slower than the unpatched route on the measured workloads, as slide 11 shows.

On the slide: tensor4all-rs PR 1278; design notes treetn-contraction-outcome.md, tree-patching-sibling-merge.md, tree-patching-elementwise.md. Exhaustive checks (R ≤ 7) meet the L2 contract. Roadmap: Grosso et al., arXiv:2602.22372.

Sources:

- https://github.com/tensor4all/tensor4all-rs/pull/1278
- https://github.com/tensor4all/tensor4all-rs/blob/main/docs/design/tree-adaptive-patching-roadmap.md
- https://arxiv.org/abs/2602.22372

## 5. Patched Fourier transform

[About 45 seconds]
patched_fourier_transform takes a patched state and returns the scheduler’s lazy result, without merging the input patches globally. The per-axis Fourier trains are embedded so the product is exact; for two interleaved axes the operator bond is 144. Applying it naively gave every bond input bond times operator bond, 7,200 at R = 10, which ran out of memory; a zip-up keeps every block bounded. For R = 20 the application truncates during its sweep and charges a proved bound, with a memory gate checked before anything is allocated. Inputs whose patches constrain the spectator indices differently are accepted. All of that is implemented. The route as a whole is marked “to optimize”: at R = 12 it is 2.1 times slower than the unpatched transform.

On the slide: tensor4all-rs PRs 1277, 1287, 1296, 1330; issues 1279, 1286, 1299, 1364.

Sources:

- https://github.com/tensor4all/tensor4all-rs/pull/1277
- https://github.com/tensor4all/tensor4all-rs/pull/1287
- https://github.com/tensor4all/tensor4all-rs/pull/1296
- https://github.com/tensor4all/tensor4all-rs/pull/1330
- https://github.com/tensor4all/tensor4all-rs/issues/1279
- https://github.com/tensor4all/tensor4all-rs/issues/1286
- https://github.com/tensor4all/tensor4all-rs/issues/1299
- https://github.com/tensor4all/tensor4all-rs/issues/1364

## 6. Merging within the error budget

[About 1 minute]
Merging patches back is where the error budget matters. In a two-stage merge, each sub-domain is merged under its own tolerance, then everything once more under the global one. Stage 2 needs a bound per stage-1 patch. Before, the only bound available was the stage-1 tolerance itself, and where it exceeded the global tolerance the patch blocked all merges above it: 23 patches instead of 13 at chi 64, and 23 instead of 1 without a cap. Implemented: the merge now returns the bound it accounted for each region, and the blocking disappears. In progress: the tolerance-driven merge policy is in review; it stores 712 parameters in one term instead of 4,296 in nine, at 1.6 to 2.4 times the scheduler time, inside the agreed 10x limit. Planned: the same bounds for merge_siblings, deferred until a caller needs them.

On the slide: U = 0 2D Hubbard Green’s function, β = 80, 64 sub-domains. tensor4all-rs issue 1367, PR 1369; PR 1407 (open); issue 1443. Also implemented: a memory limit on group sums (PR 1312).

Sources:

- https://github.com/tensor4all/tensor4all-rs/issues/1367
- https://github.com/tensor4all/tensor4all-rs/pull/1369
- https://github.com/tensor4all/tensor4all-rs/pull/1407
- https://github.com/tensor4all/tensor4all-rs/pull/1312
- https://github.com/tensor4all/tensor4all-rs/issues/1443

## 7. Sums, compression and lookups

[About 1 minute]
Three supporting pieces, all implemented. Summing many networks: patched algebra ends in sums of many terms, and an add chain re-copies the growing result each time. The direct sum now copies blocks in runs, and an N-ary sum builds the result in one pass; it is opt-in, falls back to the binary chain above a size cap, and covers f32 and complex f32. The default summing path is unchanged, so callers choose. Randomized SVD and QR: a fixed-rank rSVD with a Gaussian range finder, the rQR step, in core, usable from Fit and zip-up. Both may move to tenferro; that is not decided yet. Patch lookups: contract only the output patches you ask for, evaluate a partition at many points, and find overlapping projectors with a trie instead of a quadratic scan.

On the slide: tensor4all-rs PRs 1113–1115, 1117 (issue 1107); 1116, 1118 (issue 1108); 1076, 1234, 1295.

Sources:

- https://github.com/tensor4all/tensor4all-rs/issues/1107
- https://github.com/tensor4all/tensor4all-rs/pull/1113
- https://github.com/tensor4all/tensor4all-rs/pull/1114
- https://github.com/tensor4all/tensor4all-rs/pull/1115
- https://github.com/tensor4all/tensor4all-rs/pull/1117
- https://github.com/tensor4all/tensor4all-rs/issues/1108
- https://github.com/tensor4all/tensor4all-rs/pull/1116
- https://github.com/tensor4all/tensor4all-rs/pull/1118
- https://github.com/tensor4all/tensor4all-rs/pull/1076
- https://github.com/tensor4all/tensor4all-rs/pull/1234
- https://github.com/tensor4all/tensor4all-rs/pull/1295

## 8. Patched inputs from interpolation

[About 30 seconds]
Patched algebra needs patched inputs. Implemented: the interpolative QTT constructors take complex function values, and dyadic patches are built on caller-owned lanes, each box interpolated and compressed, split breadth first when its error estimate or rank is too large. The tutorial goes from a complex 2D construction to TreeTN to the patched Fourier transform. SimpleTT interoperability became opt-in so the tree crates no longer pull in the chain stack. On hold: extending patch parallelism to MPI; the design was approved, but the work is paused.

On the slide: tensor4all-rs PRs 1302, 1307, 1313, 1314, 1345 (issues 1276, 1335); issue 1342 (on hold).

Sources:

- https://github.com/tensor4all/tensor4all-rs/pull/1302
- https://github.com/tensor4all/tensor4all-rs/pull/1307
- https://github.com/tensor4all/tensor4all-rs/pull/1313
- https://github.com/tensor4all/tensor4all-rs/pull/1314
- https://github.com/tensor4all/tensor4all-rs/pull/1345
- https://github.com/tensor4all/tensor4all-rs/issues/1276
- https://github.com/tensor4all/tensor4all-rs/issues/1335
- https://github.com/tensor4all/tensor4all-rs/issues/1342

## 9. TreeTCI: no false “converged”

[About 1 minute]
The TreeTCI theme was false convergence. Simon’s reproducer: a chain with one site of local dimension 1 reported Converged with error zero while the true error was 0.15. Random global-search starts landed where the function is zero and measured nothing. Implemented: the recorded error now includes what the global search measured, in TreeTCI and in chain TCI; when the walks find nothing, the search probes from retained pivots and restarts subtrees; a dimension-1 vertex offers two-site candidates; and a third stage enumerates small leaf sides exhaustively, aimed at a box function where 9 of 11 seeds had stopped at error 0.75 to 0.875. In progress, as a draft: a tighter local threshold on the sampled stop path. Planned: two junction cases where the x and z bonds lock each other below the true rank. The recorded error is still a sampled diagnostic, not a bound.

On the slide: tensor4all-rs issues 1352, 1358, 1433; PRs 1356, 1357, 1361, 1365, 1434, 1440; PR 1439 (draft); issues 1435, 1436. The recorded error stays a sampled diagnostic, not a bound.

Sources:

- https://github.com/tensor4all/tensor4all-rs/issues/1352
- https://github.com/tensor4all/tensor4all-rs/issues/1358
- https://github.com/tensor4all/tensor4all-rs/issues/1433
- https://github.com/tensor4all/tensor4all-rs/pull/1356
- https://github.com/tensor4all/tensor4all-rs/pull/1357
- https://github.com/tensor4all/tensor4all-rs/pull/1361
- https://github.com/tensor4all/tensor4all-rs/pull/1365
- https://github.com/tensor4all/tensor4all-rs/pull/1434
- https://github.com/tensor4all/tensor4all-rs/pull/1440
- https://github.com/tensor4all/tensor4all-rs/pull/1439
- https://github.com/tensor4all/tensor4all-rs/issues/1435
- https://github.com/tensor4all/tensor4all-rs/issues/1436

## 10. TreeACI: ranks that settle

[About 1 minute]
TreeACI had the opposite problem: runs that never settled. On an edge whose rank is set by the threshold, pivots near it entered and left from pass to pass, so the per-edge stability window never filled and the run ended at its sweep limit. Implemented: rank hysteresis keeps the previous rank while pivots stay above a quarter of the threshold, from pass 3 on; every pivot main keeps is still kept. On the 24-case G0 gate, converged by sweep 50 went from 6 to 24, the median L2 error dropped by a third, and no case got worse. Also implemented: Guard’s query candidates are kept apart from the selected output basis, a structural fix; one seed is still less accurate. Planned: a junction case where the x and z bonds lock each other. The remaining ratchet effect of the floor is documented as a known limitation.

On the slide: G0, tt_nblock, T = 0.01, R = 9 and 10, 12 seeds each; median over K = 10…50 against main. tensor4all-rs PRs 1344 (issue 784), 1292 (issue 1265), 1348 (issue 1343); issue 1435.

Sources:

- https://github.com/tensor4all/tensor4all-rs/pull/1344
- https://github.com/tensor4all/tensor4all-rs/issues/784
- https://github.com/tensor4all/tensor4all-rs/pull/1292
- https://github.com/tensor4all/tensor4all-rs/issues/1265
- https://github.com/tensor4all/tensor4all-rs/pull/1348
- https://github.com/tensor4all/tensor4all-rs/issues/1343
- https://github.com/tensor4all/tensor4all-rs/issues/1435

## 11. Implemented, not yet efficient

[About 1 minute]
This slide is about the “to optimize” label. The patched routes are implemented and correct on their checks, but they do not pay yet, and I want to say that plainly. Momentum Fourier transform of the 2D Matsubara G0 at T = 0.01: at R = 12 and matched accuracy the patched route is 2.1 times slower for the transform and 2.8 times end to end, though it peaks at about two thirds of the memory. The per-patch transform alone is faster, but the pairwise sum of the 164 overlapping images adds more than it saves. At R = 10 none of the three library routes finished in 12 GB and an hour. For element-wise products, patching is 3 to 42 times slower on the comb tree; on tt_nblock it stores less than half the parameters but takes 20 to 36 times as long. So these routes are not adopted for these workloads until the sum can be avoided.

On the slide: One thread, ai-run measure, CLEAN runs; seeds 42, 2026, 7. tensor4all-rs issue 1364, PRs 1353, 1360; M6c: PR 1278 (exploratory, no paired protocol).

Sources:

- https://github.com/tensor4all/tensor4all-rs/issues/1364
- https://github.com/tensor4all/tensor4all-rs/pull/1353
- https://github.com/tensor4all/tensor4all-rs/pull/1360
- https://github.com/tensor4all/tensor4all-rs/pull/1278

## 12. Remaining work

[About 45 seconds]
What remains, by status. To optimize, and the biggest gap: the patched routes are implemented and correct, but cost more than they save on the workloads we measured, mainly because of the pairwise sum of overlapping images; that has to go before they are worth using. In progress: the tolerance merge policy in review and the TreeTCI sampled-stop threshold as a draft. Planned: per-region bounds for merge_siblings once a caller needs them, and the junction rank locks in TreeTCI and TreeACI. On hold: MPI patch parallelism for interpolative QTT; the design was approved, but the work is paused. In short: the algebra is implemented; making it efficient is the open question.

On the slide: To optimize: tensor4all-rs issue 1364. In progress: PRs 1407, 1439. Planned: issues 1443, 1435, 1436. On hold: issue 1342.

Sources:

- https://github.com/tensor4all/tensor4all-rs/issues/1364
- https://github.com/tensor4all/tensor4all-rs/pull/1407
- https://github.com/tensor4all/tensor4all-rs/pull/1439
- https://github.com/tensor4all/tensor4all-rs/issues/1443
- https://github.com/tensor4all/tensor4all-rs/issues/1435
- https://github.com/tensor4all/tensor4all-rs/issues/1436
- https://github.com/tensor4all/tensor4all-rs/issues/1342
