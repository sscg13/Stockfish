# Discrete halfmove embedding builds

`make rule50-ft-build`, `make rule50-hidden1-build`, and
`make rule50-hidden2-build` select `RULE50_LAYER=1`, `2`, and `3` respectively.
Pass `ARCH=native` (or an explicit non-universal architecture) and optionally `-j4`.
The default `RULE50_LAYER=0` keeps the baseline network format.

The trainer must use the matching `--rule50 ft/hidden1/hidden2` mode. All variants
use 101 rows, clamped at clock 100, and distinct file hashes. FT adds a shared
int16 row before each perspective's activation. Hidden variants add an int32
row per material stack before the selected layer's activations. `hidden1` also
conditions the two direct skip units. Clock contributions are excluded from
persistent board accumulators and caches.

These targets use non-PGO builds because the embedded baseline net is incompatible
with the experimental architecture. Set EvalFile to the matching experimental
net before evaluating or searching. Keep comparison engines non-PGO too. Clean
objects before changing `RULE50_LAYER` manually; the named targets do this.

For cache validation, build `rule50-cache-test` with the same ARCH and
RULE50_LAYER as the existing objects, then run it with an absolute experimental
net path. It checks move/undo, lazy updates, null moves, cache reuse and fresh
evaluations. AVX2 parity and search benches passed for all three modes. Cache
tests passed 18440 comparisons for hidden2 AVX2 and FT scalar. CUDA and Elo were
not tested. Existing external rule-50 damping is unchanged.
