#!/usr/bin/env bash
# Convert all .pdf figures exported by the paper_* notebooks to .eps
# Usage: ./create_eps.sh [source_dir] [output_dir]
#   defaults: .local/pdf -> .local/eps (relative to this script)
# Requires pdftops (poppler): brew install poppler

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${1:-$SCRIPT_DIR/.local/pdf}"
OUT_DIR="${2:-$SCRIPT_DIR/.local/eps}"
# resolution used only for areas that EPS cannot represent as vectors (semi-transparent fills)
RASTER_DPI=600

command -v pdftops >/dev/null || { echo "pdftops not found (brew install poppler)"; exit 1; }

mkdir -p "$OUT_DIR"

shopt -s nullglob
pdf_files=("$SRC_DIR"/*.pdf)

if [ ${#pdf_files[@]} -eq 0 ]; then
    echo "No .pdf files found in $SRC_DIR"
    exit 1
fi

for pdf in "${pdf_files[@]}"; do
    filename=$(basename "$pdf" .pdf)
    eps="$OUT_DIR/${filename}.eps"
    echo "Converting: $pdf -> $eps"
    pdftops -eps -level3 -r "$RASTER_DPI" "$pdf" "$eps"
done

echo "Done. Converted ${#pdf_files[@]} file(s) to $OUT_DIR"
