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
mkdir -p "$OUT_DIR/LICENSES/cpp-external"

cp "$BUILD_DIR/katago" "$OUT_DIR/katago"
cp "$LIBZIP_PREFIX/lib/libzip.5.dylib" "$OUT_DIR/libzip.5.dylib"
cp "$ROOT_DIR/LICENSE" "$OUT_DIR/LICENSES/KataGo-KataGomo-LICENSE.txt"
cp "$ROOT_DIR/CONTRIBUTORS" "$OUT_DIR/LICENSES/CONTRIBUTORS.txt"
cp "$ROOT_DIR/THIRD_PARTY_NOTICES.md" "$OUT_DIR/THIRD_PARTY_NOTICES.md"
cp "$LIBZIP_PREFIX/LICENSE" "$OUT_DIR/LICENSES/libzip-LICENSE.txt"
cp "$ROOT_DIR/cpp/core/sha2.cpp" "$OUT_DIR/LICENSES/sha2.cpp"

while IFS= read -r license_file; do
  rel_path="${license_file#"$ROOT_DIR/cpp/external/"}"
  rel_dir="$(dirname "$rel_path")"
  mkdir -p "$OUT_DIR/LICENSES/cpp-external/$rel_dir"
  cp "$license_file" "$OUT_DIR/LICENSES/cpp-external/$rel_path"
done < <(find "$ROOT_DIR/cpp/external" -type f \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'README*' \) | sort)

current_libzip_ref="$(otool -L "$OUT_DIR/katago" | awk '/libzip\\.5\\.dylib/{print $1; exit}')"
if [[ -n "$current_libzip_ref" && "$current_libzip_ref" != "@executable_path/libzip.5.dylib" ]]; then
  install_name_tool -change "$current_libzip_ref" "@executable_path/libzip.5.dylib" "$OUT_DIR/katago"
fi

codesign --force --sign - "$OUT_DIR/libzip.5.dylib" >/dev/null 2>&1 || true
codesign --force --sign - "$OUT_DIR/katago" >/dev/null 2>&1 || true

xattr -cr "$OUT_DIR" >/dev/null 2>&1 || true

rm -f "$ZIP_PATH"
(
  cd "$(dirname "$OUT_DIR")"
  COPYFILE_DISABLE=1 /usr/bin/zip -r -X -q "$ZIP_PATH" "$(basename "$OUT_DIR")"
)

echo "Packaged binary directory: $OUT_DIR"
echo "Packaged zip asset:       $ZIP_PATH"
