#!/bin/bash
# Board-lint gate: PROJECT_RULES.md rule 6. Parses PATHWAY_FORWARD.md's
# "## Rows" table and asserts (a) every data row has exactly 4
# `|`-delimited cells (code spans masked first, so a literal `|` inside
# backticks — e.g. a shell pipe in an Evidence command, or `` `|` `` itself
# — is not mistaken for a column delimiter), and (b) every Status cell
# starts with one of the declared vocabulary tokens
# (TODO / DOING / DONE / BLOCKED(owner)), optionally followed by free text.
# Documented, pre-existing exceptions are WARNed, not failed — see rule 6.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - "PATHWAY_FORWARD.md" <<'PYEOF'
import re, sys

path = sys.argv[1]
lines = open(path).read().splitlines()

# \b fails right after the literal ')' in BLOCKED(owner) at end-of-string
# (')' is a non-word char, so there is no word boundary there) — match the
# token then require either end-of-string or a non-alnum separator instead.
VOCAB = re.compile(r'^(TODO|DOING|DONE|BLOCKED\(owner\))($|[^A-Za-z0-9])')

# Documented exceptions to the Status-vocabulary check (PROJECT_RULES.md
# rule 6): pre-existing rows whose Status predates this rule and are not
# being silently rewritten by the lint script that introduces the rule.
EXCEPTIONS = {
    16: "legacy 'PARTIAL, not TODO (...)' status predates rule 6; "
        "plan-vs-code correction note, left as written rather than reworded",
}


def mask_backtick_pipes(line):
    """Replace `|` chars that fall inside a backtick code span with \\x00
    so splitting on `|` does not treat them as column delimiters."""
    out = []
    in_code = False
    for c in line:
        if c == '`':
            in_code = not in_code
            out.append(c)
        elif c == '|' and in_code:
            out.append('\x00')
        else:
            out.append(c)
    return ''.join(out)


in_rows = False
header_seen = False
sep_seen = False
fail = False
warn_rows = []
row_count = 0

for lineno, line in enumerate(lines, 1):
    if line.startswith('## Rows'):
        in_rows = True
        continue
    if not in_rows or not line.startswith('|'):
        continue
    if not header_seen:
        header_seen = True
        continue
    if not sep_seen:
        sep_seen = True
        continue

    masked = mask_backtick_pipes(line)
    cells = masked.split('|')
    if cells and cells[0].strip() == '':
        cells = cells[1:]
    if cells and cells[-1].strip() == '':
        cells = cells[:-1]
    cells = [c.replace('\x00', '|') for c in cells]

    if len(cells) != 4:
        print(f"BOARD LINT FAIL: {path}:{lineno} has {len(cells)} "
              f"|-delimited cells, expected 4")
        fail = True
        continue

    row_count += 1
    rownum_raw = cells[0].strip()
    status = cells[2].strip()
    try:
        rownum = int(rownum_raw)
    except ValueError:
        print(f"BOARD LINT FAIL: {path}:{lineno} row id {rownum_raw!r} "
              f"is not an integer")
        fail = True
        continue

    if not VOCAB.match(status):
        if rownum in EXCEPTIONS:
            warn_rows.append((rownum, lineno, EXCEPTIONS[rownum]))
        else:
            print(f"BOARD LINT FAIL: {path}:{lineno} row {rownum} status "
                  f"{status[:70]!r} does not start with "
                  f"TODO/DOING/DONE/BLOCKED(owner)")
            fail = True

for rownum, lineno, reason in warn_rows:
    print(f"BOARD LINT WARN: {path}:{lineno} row {rownum} documented "
          f"exception (rule 6) — {reason}")

if row_count == 0:
    print(f"BOARD LINT FAIL: no data rows parsed under '## Rows' in {path}")
    fail = True

print(f"board lint: {row_count} rows checked, {len(warn_rows)} documented "
      f"exception(s)")
sys.exit(1 if fail else 0)
PYEOF
