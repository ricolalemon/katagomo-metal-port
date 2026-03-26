# v0.1.0

First public release of the Gomoku Metal forward-port.

## Highlights

* Forward-port of KataGomo Gomoku-specific logic onto KataGo `v1.16.4`
* Apple Silicon Metal backend build path
* macOS 15 packaging flow with bundled `libzip.5.dylib`
* Smoke-tested on Apple Silicon macOS `15.7.5` and `26.4`
* Gomoku sample release config included
* Benchmark and regression notes included in the repository

## Important Notes

* This is an unofficial fork, not an upstream KataGo or KataGomo release.
* The tested release target is Apple Silicon on macOS 15.x.
* Metal and OpenCL are not bit-identical; some positions can still diverge.

## Recommended Starting Points

* Build and packaging instructions: `docs/GomokuMetalPort.md`
* Benchmark summary: `docs/GomokuMetalBenchmarks.md`
* Sample config: `cpp/configs/gomoku_freestyle_metal_release.cfg`
