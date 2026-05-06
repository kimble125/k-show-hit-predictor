#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Desktop/IT/kimble125/k-show-hit-predictor}"
CSV_SRC="${CSV_SRC:-$HOME/Library/CloudStorage/GoogleDrive-hoykim125@gmail.com/내 드라이브/Git/k-drama-hit-predictor/output/nielsen_weekly_all_categories_2024_2026.csv}"
CSV_DST="$REPO_DIR/data/nielsen_weekly_all_categories_2024_2026.csv"

echo "Repository: $REPO_DIR"
echo "CSV source: $CSV_SRC"
echo "CSV target: $CSV_DST"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Repository directory not found. Clone it first:"
  echo "git clone https://github.com/kimble125/k-show-hit-predictor.git \"$REPO_DIR\""
  exit 1
fi

if [ ! -f "$CSV_SRC" ]; then
  echo "CSV source file not found. Check CSV_SRC path."
  exit 1
fi

mkdir -p "$REPO_DIR/data"
cp "$CSV_SRC" "$CSV_DST"

cd "$REPO_DIR"
git add data/nielsen_weekly_all_categories_2024_2026.csv
git commit -m "data: add Nielsen weekly ratings CSV" || echo "No CSV changes to commit."
git push origin main

echo "Done."
