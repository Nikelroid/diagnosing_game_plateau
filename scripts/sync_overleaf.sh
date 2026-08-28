#!/bin/bash
# Push the paper to Overleaf. Run this after any change to paper/, figures/ or tables/.
#
#   scripts/sync_overleaf.sh            # push the workshop paper
#   scripts/sync_overleaf.sh --check    # report drift, push nothing
#
# Overleaf wants a self-contained project, so ../figures and ../tables are rewritten to local
# subdirectories and only the assets this paper actually inputs are copied. The push is refused if
# the paper does not compile or runs past its page limit, because a broken Overleaf project is
# worse than a stale one.
set -uo pipefail

PROJECT=6a9187545883feb1b5e78337     # "Is it the game or the agent?" (IAEval workshop)
LIMIT=4                              # content pages, excluding references
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=$SCRATCH/envs/coev/bin/python
CHECK_ONLY=${1:-}
redact() { sed -E 's/olp_[A-Za-z0-9]+/olp_REDACTED/g; s|git:[^@]*@|git:REDACTED@|g'; }

B=$(mktemp -d $SCRATCH/ol_bundle_XXXX)
trap 'rm -rf "$B" "${C:-}"' EXIT
mkdir -p "$B/figures" "$B/tables"
sed -e 's|{{\.\./figures/}}|{{figures/}}|' -e 's|\.\./tables/|tables/|' "$SRC/paper/main.tex" > "$B/main.tex"
cp "$SRC/paper/refs.tex" "$B/"
# Copy every asset the paper references, and fail loudly if one is missing. A silently absent
# figure still compiles under nonstopmode and still reports the right page count, so without this
# check a broken paper can pass every other gate and reach Overleaf.
missing=0
for f in $(grep -oE "includegraphics\[[^]]*\]\{[^}]+\}" "$B/main.tex" | grep -oE "\{[^}]+\}$" | tr -d '{}'); do
  if   [ -f "$SRC/figures/$f" ];     then cp "$SRC/figures/$f" "$B/figures/"
  elif [ -f "$SRC/figures/$f.pdf" ]; then cp "$SRC/figures/$f.pdf" "$B/figures/"
  else echo "  MISSING FIGURE: $f"; missing=1
  fi
done
for t in $(grep -oE "input\{tables/[^}]+\}" "$B/main.tex" | sed 's|input{tables/||; s|}||'); do
  if [ -f "$SRC/tables/$t" ]; then cp "$SRC/tables/$t" "$B/tables/"
  else echo "  MISSING TABLE: $t"; missing=1
  fi
done
[ "$missing" -eq 0 ] || { echo "referenced assets are missing; nothing pushed"; exit 1; }

module load texlive/2026 2>/dev/null
( cd "$B" && for i in 1 2 3; do pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1; done )
[ -f "$B/main.pdf" ] || { echo "paper does not compile; nothing pushed"; exit 1; }

read -r PAGES BAD <<<"$(PYTHONNOUSERSITE= $PY -c "
import fitz
d = fitz.open('$B/main.pdf')
ref = next(i+1 for i,p in enumerate(d) if 'References' in p.get_text())
print(ref-1, sum(p.get_text().count('[?]') for p in d))")"
echo "  content pages: ${PAGES}/${LIMIT}   unresolved citations: ${BAD}"
if ! [ "${PAGES:-x}" -eq "${PAGES:-x}" ] 2>/dev/null; then echo "page check failed; refusing"; exit 2; fi
[ "$PAGES" -gt "$LIMIT" ] && { echo "over the page limit; refusing to push"; exit 2; }
[ "$BAD" -gt 0 ] && { echo "unresolved citations; refusing to push"; exit 2; }

TOKEN=$(bash -lc 'source ~/.bashrc >/dev/null 2>&1; printf "%s" "$OVERLEAF_TOKEN"')
[ -n "$TOKEN" ] || { echo "no OVERLEAF_TOKEN"; exit 3; }
C=$(mktemp -d $SCRATCH/ol_clone_XXXX); rm -rf "$C"
git clone -q "https://git:${TOKEN}@git.overleaf.com/${PROJECT}" "$C" 2>&1 | redact

rm -rf "$C"/figures "$C"/tables
cp -r "$B"/main.tex "$B"/refs.tex "$B"/figures "$B"/tables "$C"/
rm -f "$C"/*.aux "$C"/*.log "$C"/*.out "$C"/*.pdf "$C"/*.synctex.gz
git -C "$C" add -A
if git -C "$C" diff --cached --quiet; then
  echo "  Overleaf already up to date"; exit 0
fi
if [ "$CHECK_ONLY" = "--check" ]; then
  echo "  DRIFT: Overleaf differs from local (run without --check to push)"
  git -C "$C" diff --cached --stat | tail -5; exit 0
fi
git -C "$C" -c user.name="Nima Kelidari" \
    -c user.email="68930046+Nikelroid@users.noreply.github.com" \
    commit -q -m "sync from repo: $(cd "$SRC" && git log -1 --format=%h) $(date -u +%Y-%m-%d)"
git -C "$C" push -q origin "$(git -C "$C" rev-parse --abbrev-ref HEAD)" 2>&1 | redact
echo "  pushed to Overleaf project ${PROJECT}"
