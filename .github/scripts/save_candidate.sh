#!/usr/bin/env bash
# Guarda arquivos gerados no workflow (um modelo candidato e o resumo dele) numa branch órfã, só com eles, para serem
# conferidos antes de entrar no sistema. Se algo falhar, a anotação mostra o comando e a mensagem exata.
#
#     .github/scripts/save_candidate.sh <branch> <arquivo>...
set -u
branch=$1
shift

run() {
  if ! out=$("$@" 2>&1); then
    echo "::error title=Falhou: $*::$(echo "$out" | tr '\n' ' ' | cut -c1-800) | disco: $(df -h . | tail -1)"
    exit 1
  fi
}

run git config user.name "github-actions[bot]"
run git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
run git checkout --orphan "$branch"
run git rm -rq --cached .
for file in "$@"; do
  run git add -f "$file"
done
run git commit -q -m "Candidato gerado pelo workflow \"$GITHUB_WORKFLOW\" (${GITHUB_SHA::7})"
for attempt in 1 2 3; do
  if git push -f origin "$branch" 2> push.err; then
    echo "::notice title=Candidato guardado::branch $branch"
    exit 0
  fi
  sleep $((15 * attempt))
done
echo "::error title=Push falhou::$(tr '\n' ' ' < push.err | cut -c1-800)"
exit 1
