#!/usr/bin/env bash
#
# Publish wiki/ to the GitHub wiki repository.
#
# The GitHub wiki is a separate git repository (<repo>.wiki.git) that GitHub
# only creates once the wiki has been enabled and at least one page exists.
# If this script reports "Repository not found", open
# https://github.com/SamarjitDebnath/ephemeris-serve/wiki and create any page
# once (Settings -> Features -> Wikis must also be checked), then re-run.
#
# The wiki repo is treated as a publish target, not a source of truth: pages
# under wiki/ overwrite whatever is there. Edit wiki/ in the main repo instead
# of editing pages in the GitHub wiki UI.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIKI_SRC="${REPO_ROOT}/wiki"
WIKI_REMOTE="${WIKI_REMOTE:-https://github.com/SamarjitDebnath/ephemeris-serve.wiki.git}"
WORK_DIR="$(mktemp -d)"

cleanup() { rm -rf "${WORK_DIR}"; }
trap cleanup EXIT

if [ ! -d "${WIKI_SRC}" ]; then
  echo "error: ${WIKI_SRC} does not exist" >&2
  exit 1
fi

echo "==> Cloning ${WIKI_REMOTE}"
if ! git clone --quiet "${WIKI_REMOTE}" "${WORK_DIR}/wiki"; then
  cat >&2 <<'EOF'

error: could not clone the wiki repository.

GitHub creates <repo>.wiki.git only after the wiki has been enabled and one
page exists. Fix it once:

  1. Repository Settings -> Features -> check "Wikis"
  2. Open https://github.com/SamarjitDebnath/ephemeris-serve/wiki
  3. Click "Create the first page", save anything (it gets overwritten)
  4. Re-run: make wiki-sync
EOF
  exit 1
fi

echo "==> Copying pages"
find "${WORK_DIR}/wiki" -maxdepth 1 -name '*.md' -delete
cp "${WIKI_SRC}"/*.md "${WORK_DIR}/wiki/"

cd "${WORK_DIR}/wiki"
git add -A

if git diff --cached --quiet; then
  echo "==> Wiki already up to date"
  exit 0
fi

git commit --quiet -m "Sync wiki from ${REPO_ROOT##*/}@$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
git push --quiet origin HEAD
echo "==> Pushed: https://github.com/SamarjitDebnath/ephemeris-serve/wiki"
