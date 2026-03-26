# Gomoku Metal Port

This repository contains a local forward-port of KataGomo's Gomoku-specific changes onto KataGo `v1.16.4`, plus a working Apple Silicon Metal build and packaging path for macOS 15.

## Scope

What this fork adds or preserves:

* Gomoku protocol entrypoints such as `gomprotocol`
* Gomoku board/rules/search helpers from the KataGomo line
* Apple Silicon Metal backend bring-up on top of the `v1.16.4` KataGo codebase
* A macOS 15 packaging flow that fixes deployment target and `libzip` runtime linking

What this fork is not:

* Not an official KataGo release
* Not an official KataGomo release
* Not yet upstreamed

## Tested Environment

The release path described below has been tested on:

* Apple Silicon Macs
* macOS `15.7.4`, `15.7.5`, and `26.4`
* AppleClang + Swift from modern Xcode
* Ninja-based CMake builds

## Prerequisites

Install the following first:

* Xcode Command Line Tools or full Xcode
* `cmake`
* `ninja`
* `libzip`

On Homebrew:

```bash
brew install cmake ninja libzip
```

## Build The Metal Binary

From the repository root:

```bash
SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"

cmake -S cpp -B cpp/build-metal-port-ninja \
  -G Ninja \
  -DUSE_BACKEND=METAL \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_SYSROOT="$SDKROOT" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0

cmake --build cpp/build-metal-port-ninja --target katago
```

The output binary will be:

```text
cpp/build-metal-port-ninja/katago
```

## Build A Release Package

This repository includes a packaging script that builds the Metal binary, copies the runtime `libzip` dependency, rewrites the install name, and produces both a folder and a zip asset.

```bash
./misc/package_macos15_metal.sh
```

Default outputs:

```text
macos15-metal-package/
macos15-metal-package.zip
```

You can also specify a custom output directory:

```bash
./misc/package_macos15_metal.sh /tmp/katagomo-metal-release
```

## Run Gomoku GTP

Example:

```bash
./macos15-metal-package/katago gtp \
  -config cpp/configs/gomoku_freestyle_metal_release.cfg \
  -model /path/to/zhizi_renju28b_s1600.bin.gz
```

## Release Config Notes

The included sample config is intentionally opinionated for interactive Gomoku play on Apple Silicon:

* `numSearchThreads = 1`
* `chosenMoveTemperatureEarly = 0.0`
* `chosenMoveTemperature = 0.0`
* `searchFactorWhenWinning = 1.0`
* `searchFactorWhenWinningThreshold = 1.0`
* `allowResignation = false`

These are product choices rather than official KataGo defaults.

## Known Limitations

* This is a forward-port, not an upstream-supported branch.
* Metal and OpenCL are not bit-identical. Even with deterministic testing overrides, some positions still produce different best moves.
* The macOS release path has only been validated on Apple Silicon and on the tested macOS releases listed above.
* The repository contains Gomoku-specific work that has not been revalidated across every upstream KataGo feature area. The tested path is the Gomoku GTP engine, not the full distributed/selfplay/training surface.
* The sample release config is aimed at interactive play and analysis, not large-scale training workloads.

## Suggested Publish Strategy

For a first public fork/release:

1. Publish the source branch as your own public repository.
2. Attach the packaged `macos15-metal-package.zip` asset to a GitHub release.
3. Keep this fork clearly labeled as experimental and Apple-Silicon-focused.
4. Upstream later, after more soak testing and code cleanup.
