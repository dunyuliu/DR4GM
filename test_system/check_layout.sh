#!/bin/bash
# Layout gate: fails if a tracked root entry is outside the whitelist or a
# committed file exceeds 5 MB. Run by run_tests.sh before any scenario runs.
set -euo pipefail
cd "$(dirname "$0")/.."
git rev-parse --git-dir >/dev/null 2>&1 || { echo "layout check: not a git checkout, skipped"; exit 0; }

ALLOWED='^(README\.md|LICENSE|CITATION\.cff|\.gitignore|requirements\.txt|scripts|docs|test_system|src|data|\.github|PATHWAY_FORWARD\.md|PROJECT_RULES\.md|CLAUDE\.md|release_notes_v[0-9][0-9.a-z-]*\.md)$'
# vendored upstream test data, kept until the owner decides LFS vs fetch
SIZE_EXEMPT='^src/gmpe-smtk/tests/(file_samples/core_flatfile_ngawest2\.csv|smtk_ims_test_data\.hdf5)$'

fail=0
while read -r e; do
  [[ "$e" =~ $ALLOWED ]] || { echo "LAYOUT FAIL: tracked root entry '$e' not in whitelist"; fail=1; }
done < <(git ls-files | awk -F/ '{print $1}' | sort -u)

while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  [[ "$f" =~ $SIZE_EXEMPT ]] && continue
  s=$(stat -c %s "$f")
  [ "$s" -gt 5242880 ] && { echo "LAYOUT FAIL: $f is $s bytes (> 5 MB)"; fail=1; }
done < <(git ls-files -z)

for p in results/ reference/ local/; do
  git check-ignore -q "$p" || { echo "LAYOUT FAIL: $p is not git-ignored"; fail=1; }
done

[ "$fail" -eq 0 ] && echo "layout check: PASS"
exit "$fail"
