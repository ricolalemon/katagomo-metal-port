# Gomoku Metal Benchmarks

This page summarizes the local benchmark and regression work used to evaluate the Metal forward-port.

## Environment

Unless otherwise noted, the measurements below were run on Apple Silicon using:

* KataGomo-style Gomoku port on top of KataGo `v1.16.4`
* `zhizi_renju28b_s1600.bin.gz`
* `numSearchThreads = 1`

## Backend Speed: v1.16.4 Metal vs v1.16.4 OpenCL

Same codebase, same model, same parameters, backend only changed.

| Case | Metal | OpenCL | Takeaway |
| --- | ---: | ---: | --- |
| `56 visits` | `0.6267s` | `1.4431s` | Metal about `2.3x` faster |
| high-visits batch wall time | `6.08s` | `6.21s` | wall time close, but Metal searched much deeper |
| high-visits batch actual visits | `760.33` | `297.33` | Metal delivered about `2.6x` more search under similar time |

## Deterministic Backend A/B

Same `v1.16.4` codebase, same model, deterministic overrides enabled to remove seed noise.

| Metric | Result |
| --- | ---: |
| Fixed positions tested | `22` |
| OpenCL self-repeat stable | `22/22` |
| Metal self-repeat stable | `22/22` |
| Exact same best move across backends | `16/22` |

Interpretation:

* Backend differences remain even after removing randomness.
* The Metal port should not be described as "bit-identical to OpenCL".
* The observed differences are stable, not random noise.

## Old Production OpenCL vs New Metal Regression Check

Comparison:

* old production-style OpenCL branch based on the older KataGomo line
* new Metal port on top of `v1.16.4`

### Mixed case set

| Metric | Result |
| --- | ---: |
| positions | `22` |
| same move | `17/22` |
| divergent positions | `5` |
| referee judged new better | `1` |
| referee judged old better | `0` |
| referee judged tied | `4` |

### Official Gomocup opening roots

| Metric | Result |
| --- | ---: |
| opening root cases | `12` |
| same move | `12/12` |

Interpretation:

* The new Metal port did not show evidence of a strength regression in this check.
* It looked slightly better or at least not weaker, but not dramatically stronger.

## Pure-Visits Strength Sweeps On Metal

All matches below used Gomocup `freestyle15` openings, first four openings, colors swapped, `8` games total per sweep.

### `56 vs 200`

| Metric | Result |
| --- | ---: |
| `56` wins | `0` |
| `200` wins | `3` |
| draws | `5` |
| avg move time @ `56` | `430.6ms` |
| avg move time @ `200` | `1376.0ms` |

### `200 vs 1000`

| Metric | Result |
| --- | ---: |
| `200` wins | `0` |
| `1000` wins | `5` |
| draws | `3` |
| avg move time @ `200` | `1216.9ms` |
| avg move time @ `1000` | `5078.8ms` |

### `1000 vs 5000`

| Metric | Result |
| --- | ---: |
| `1000` wins | `1` |
| `5000` wins | `2` |
| draws | `5` |
| avg move time @ `1000` | `6066.2ms` |
| avg move time @ `5000` | `26314.1ms` |

Interpretation:

* `56 -> 200` is a clear strength upgrade.
* `200 -> 1000` is another clear upgrade.
* `1000 -> 5000` shows diminishing returns relative to the large time cost.

## Parameter Style Check At `500 visits`

Comparison:

* "master-style" release config
* "official-like" temperature/search-factor config

Same model, same backend, same `500 visits`, official Gomocup openings, `8` games total.

| Metric | Result |
| --- | ---: |
| master-style wins | `2` |
| official-like wins | `2` |
| draws | `4` |
| avg move time, master-style | `2832.7ms` |
| avg move time, official-like | `2808.1ms` |

Interpretation:

* At `500 visits`, the two parameter styles looked broadly comparable in strength.
* The main practical difference is style and determinism, not a clear Elo gap.

## Practical Release Takeaways

Based on the sweeps above:

* `200 visits` is a good "fast but real" play tier.
* `1000 visits` is a strong premium tier.
* `5000 visits` is better treated as an analysis tier, not a normal play tier.
* Metal is worth shipping on Apple Silicon because it improves usable search depth per unit time.
