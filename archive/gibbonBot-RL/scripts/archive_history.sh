#!/usr/bin/env bash
# Extract every historic version of every large/binary blob ever committed
# (across ALL branches, refs, tags, stashes) into a local archive folder,
# named so collisions are impossible and duplicates collapse:
#
#   archive/git_history/<blob-sha>__<sanitized-original-path>
#
# Plus a manifest CSV: archive/git_history/_manifest.csv with
# (blob_sha, size_bytes, original_path, sha256_of_blob).
#
# Idempotent — re-running skips blobs already on disk.
#
# Usage:
#   scripts/archive_history.sh                 # extract anything >50KB
#   MIN_SIZE=1 scripts/archive_history.sh      # extract literally everything (huge!)
set -euo pipefail

cd "$(dirname "$0")/.."

ARCHIVE="${ARCHIVE:-/home/tony/gibbonBot-RL-archive}"
OUT="$ARCHIVE/git_history"
MANIFEST="$OUT/_manifest.csv"
MIN_SIZE="${MIN_SIZE:-50000}"

# File extensions we want to archive (binary/data files that don't belong in git history).
# Source code, docs, JSON config, etc. are intentionally excluded.
PATTERN='\.(npz|safetensors|db|zip|stl|png|jpg|jpeg|pem|tar|gz|tgz|pt|pth|onnx|bin|gguf|webm|mp4|mov|wav|mp3)$'

mkdir -p "$OUT"
[[ -f "$MANIFEST" ]] || echo "blob_sha,size_bytes,original_path,sha256" >"$MANIFEST"

# Build candidate list: (blob_sha, size, path) for every blob ever in history >= MIN_SIZE
# whose path matches our pattern.
echo "[archive] enumerating blobs from all refs (this can take a minute)..." >&2
candidates=$(mktemp)
trap 'rm -f "$candidates"' EXIT

git rev-list --objects --all 2>/dev/null \
    | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' 2>/dev/null \
    | awk -v min="$MIN_SIZE" -v pat="$PATTERN" '
        $1 == "blob" && $3 >= min && $4 != "" {
            path = $4
            for (i = 5; i <= NF; i++) path = path " " $i
            if (path ~ pat) print $2 "\t" $3 "\t" path
        }' \
    | sort -u >"$candidates"

total=$(wc -l <"$candidates")
echo "[archive] $total candidate blobs found" >&2

n=0; new=0; skipped=0; bytes_new=0
while IFS=$'\t' read -r sha size path; do
    n=$((n+1))
    safe_path=$(printf '%s' "$path" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')
    dest="$OUT/${sha}__${safe_path}"
    if [[ -e "$dest" ]]; then
        skipped=$((skipped+1))
    else
        git cat-file blob "$sha" >"$dest"
        h=$(sha256sum "$dest" | awk '{print $1}')
        # Escape commas/quotes in path for CSV
        csv_path=$(printf '%s' "$path" | sed 's/"/""/g')
        printf '%s,%s,"%s",%s\n' "$sha" "$size" "$csv_path" "$h" >>"$MANIFEST"
        new=$((new+1))
        bytes_new=$((bytes_new + size))
    fi
    if (( n % 50 == 0 )); then
        printf '\r[archive] %d/%d  new=%d  skipped=%d  %.1f MB extracted' \
            "$n" "$total" "$new" "$skipped" "$(awk -v b=$bytes_new 'BEGIN{print b/1048576}')" >&2
    fi
done <"$candidates"

printf '\n[archive] done. %d blobs new (%.1f MB), %d already present.\n' \
    "$new" "$(awk -v b=$bytes_new 'BEGIN{print b/1048576}')" "$skipped" >&2
echo "[archive] manifest: $MANIFEST" >&2
echo "[archive] total archive size:" >&2
du -sh "$OUT" >&2
