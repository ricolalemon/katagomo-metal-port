#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/cpp/build-metal-port-ninja}"
OUT_DIR="${1:-$ROOT_DIR/macos15-metal-package}"
if [[ -n "${ZIP_PATH:-}" ]]; then
  ZIP_PATH="$ZIP_PATH"
elif [[ $# -ge 1 ]]; then
  ZIP_PATH="${OUT_DIR}.zip"
else
  ZIP_PATH="$ROOT_DIR/macos15-metal-package.zip"
fi

SDKROOT="${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}"
LIBZIP_PREFIX="${LIBZIP_PREFIX:-$(brew --prefix libzip 2>/dev/null || true)}"

if [[ -z "$LIBZIP_PREFIX" || ! -f "$LIBZIP_PREFIX/lib/libzip.5.dylib" ]]; then
  echo "libzip.5.dylib not found. Install it with: brew install libzip" >&2
  exit 1
fi

cmake -S "$ROOT_DIR/cpp" -B "$BUILD_DIR" \
  -G Ninja \
  -DUSE_BACKEND=METAL \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_SYSROOT="$SDKROOT" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0

cmake --build "$BUILD_DIR" --target katago

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

cp "$BUILD_DIR/katago" "$OUT_DIR/katago"
cp "$LIBZIP_PREFIX/lib/libzip.5.dylib" "$OUT_DIR/libzip.5.dylib"

current_libzip_ref="$(otool -L "$OUT_DIR/katago" | awk '/libzip\\.5\\.dylib/{print $1; exit}')"
if [[ -n "$current_libzip_ref" && "$current_libzip_ref" != "@executable_path/libzip.5.dylib" ]]; then
  install_name_tool -change "$current_libzip_ref" "@executable_path/libzip.5.dylib" "$OUT_DIR/katago"
fi

codesign --force --sign - "$OUT_DIR/libzip.5.dylib" >/dev/null 2>&1 || true
codesign --force --sign - "$OUT_DIR/katago" >/dev/null 2>&1 || true

rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$OUT_DIR" "$ZIP_PATH"

echo "Packaged binary directory: $OUT_DIR"
echo "Packaged zip asset:       $ZIP_PATH"
