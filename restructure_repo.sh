#!/usr/bin/env bash

set -e

echo "========================================"
echo " Enterprise AI Platform Restructure"
echo "========================================"

# ------------------------------------------------------------
# 1. Safety checks
# ------------------------------------------------------------

if [ ! -d ".git" ]; then
  echo "ERROR: Run this script from the repository root."
  exit 1
fi

echo
echo "Current repo:"
git remote -v

echo
echo "Current branch:"
git branch --show-current

# Don't restructure with uncommitted changes.
if [ -n "$(git status --porcelain)" ]; then
  echo
  echo "ERROR: You have uncommitted changes."
  echo "Commit or stash them before running this script."
  echo
  git status --short
  exit 1
fi

# ------------------------------------------------------------
# 2. Backup branch
# ------------------------------------------------------------

BACKUP_BRANCH="backup-before-restructure-$(date +%Y%m%d-%H%M%S)"

echo
echo "Creating safety branch: $BACKUP_BRANCH"

git branch "$BACKUP_BRANCH"

# ------------------------------------------------------------
# 3. Create new top-level structure
# ------------------------------------------------------------

echo
echo "Creating target directories..."

mkdir -p learning
mkdir -p learning/foundations
mkdir -p learning/rag

mkdir -p mini-projects

mkdir -p capstone
mkdir -p capstone/docs/adr
mkdir -p capstone/data/raw
mkdir -p capstone/data/processed
mkdir -p capstone/data/corpus
mkdir -p capstone/data/golden-set
mkdir -p capstone/notebooks
mkdir -p capstone/src/enterprise_ai/config
mkdir -p capstone/src/enterprise_ai/ingestion
mkdir -p capstone/src/enterprise_ai/embeddings
mkdir -p capstone/src/enterprise_ai/retrieval
mkdir -p capstone/src/enterprise_ai/rag
mkdir -p capstone/src/enterprise_ai/agents
mkdir -p capstone/src/enterprise_ai/evaluation
mkdir -p capstone/src/enterprise_ai/observability
mkdir -p capstone/src/enterprise_ai/storage
mkdir -p capstone/src/enterprise_ai/api
mkdir -p capstone/tests/unit
mkdir -p capstone/tests/integration
mkdir -p capstone/tests/evaluation
mkdir -p capstone/scripts

mkdir -p shared/llm
mkdir -p shared/utils
mkdir -p shared/evaluation

# ------------------------------------------------------------
# 4. Move course / learning content
# ------------------------------------------------------------

echo
echo "Moving learning material..."

if [ -d "practice_demo" ]; then
  git mv practice_demo learning/practice_demo
fi

if [ -d "learner" ]; then
  mkdir -p mini-projects/mp01-prompt-lab
  git mv learner mini-projects/mp01-prompt-lab/learner
fi

# ------------------------------------------------------------
# 5. Move MP1
# ------------------------------------------------------------

echo
echo "Moving Mini Project 1..."

if [ -d "MP1" ]; then
  mkdir -p mini-projects/mp01-prompt-lab
  git mv MP1 mini-projects/mp01-prompt-lab/submission
fi

# ------------------------------------------------------------
# 6. Move root course data
# ------------------------------------------------------------

if [ -d "data" ]; then
  mkdir -p mini-projects/mp01-prompt-lab
  git mv data mini-projects/mp01-prompt-lab/data
fi

# ------------------------------------------------------------
# 7. Move existing root src into learning
# ------------------------------------------------------------

if [ -d "src" ]; then
  git mv src learning/foundations/src
fi

# ------------------------------------------------------------
# 8. Move existing root docs
# ------------------------------------------------------------

if [ -d "docs" ]; then
  git mv docs learning/course-docs
fi

# ------------------------------------------------------------
# 9. Migrate existing capstone
# ------------------------------------------------------------

echo
echo "Migrating existing knowledge-assistant-capstone..."

if [ -d "knowledge-assistant-capstone" ]; then

  # Existing practice examples belong in learning
  if [ -d "knowledge-assistant-capstone/practice_examples" ]; then
    mkdir -p learning/capstone-experiments
    git mv \
      knowledge-assistant-capstone/practice_examples \
      learning/capstone-experiments/practice_examples
  fi

  # Existing capstone src
  if [ -d "knowledge-assistant-capstone/src" ]; then
    mkdir -p capstone/legacy
    git mv \
      knowledge-assistant-capstone/src \
      capstone/legacy/src
  fi

  # Existing capstone data
  if [ -d "knowledge-assistant-capstone/data" ]; then
    mkdir -p capstone/legacy
    git mv \
      knowledge-assistant-capstone/data \
      capstone/legacy/data
  fi

  # Existing capstone docs
  if [ -d "knowledge-assistant-capstone/docs" ]; then
    mkdir -p capstone/legacy
    git mv \
      knowledge-assistant-capstone/docs \
      capstone/legacy/docs
  fi

  # Preserve existing README
  if [ -f "knowledge-assistant-capstone/README.md" ]; then
    git mv \
      knowledge-assistant-capstone/README.md \
      capstone/README-old.md
  fi

  # Preserve anything else rather than deleting it
  if [ "$(ls -A knowledge-assistant-capstone 2>/dev/null)" ]; then
    mkdir -p capstone/legacy/knowledge-assistant-capstone
    shopt -s dotglob nullglob

    for item in knowledge-assistant-capstone/*; do
      git mv "$item" capstone/legacy/knowledge-assistant-capstone/
    done

    shopt -u dotglob nullglob
  fi

  rmdir knowledge-assistant-capstone 2>/dev/null || true
fi

# ------------------------------------------------------------
# 10. Keep root requirements for now
# ------------------------------------------------------------

echo
echo "Keeping root requirements.txt as shared environment."

# ------------------------------------------------------------
# 11. Python package placeholders
# ------------------------------------------------------------

touch capstone/src/enterprise_ai/__init__.py
touch capstone/src/enterprise_ai/config/__init__.py
touch capstone/src/enterprise_ai/ingestion/__init__.py
touch capstone/src/enterprise_ai/embeddings/__init__.py
touch capstone/src/enterprise_ai/retrieval/__init__.py
touch capstone/src/enterprise_ai/rag/__init__.py
touch capstone/src/enterprise_ai/agents/__init__.py
touch capstone/src/enterprise_ai/evaluation/__init__.py
touch capstone/src/enterprise_ai/observability/__init__.py
touch capstone/src/enterprise_ai/storage/__init__.py
touch capstone/src/enterprise_ai/api/__init__.py

# Git doesn't track empty directories
touch capstone/data/raw/.gitkeep
touch capstone/data/processed/.gitkeep
touch capstone/data/corpus/.gitkeep
touch capstone/data/golden-set/.gitkeep
touch capstone/notebooks/.gitkeep
touch capstone/tests/unit/.gitkeep
touch capstone/tests/integration/.gitkeep
touch capstone/tests/evaluation/.gitkeep
touch capstone/scripts/.gitkeep
touch shared/llm/.gitkeep
touch shared/utils/.gitkeep
touch shared/evaluation/.gitkeep

# ------------------------------------------------------------
# 12. Create capstone README if it doesn't exist
# ------------------------------------------------------------

if [ ! -f "capstone/README.md" ]; then
cat > capstone/README.md <<'CAPSTONE_EOF'
# AI-Powered Document Freshness & Knowledge Governance Platform

This directory contains the production-oriented capstone implementation.

## Evolution

The project is expected to evolve through:

1. Naive RAG
2. Advanced RAG
3. RAG Evaluation
4. Agentic RAG
5. Document Freshness Agents
6. Observability and Governance

## Directory Structure

- `data/` - capstone corpus and evaluation datasets
- `notebooks/` - exploratory capstone analysis
- `src/enterprise_ai/` - reusable production code
- `tests/` - automated tests and evaluation
- `scripts/` - runnable utilities
- `docs/` - architecture and ADRs

## Rule

Experimental learning code belongs under `/learning`.

Reusable code that becomes part of the final platform belongs under `/capstone/src`.
CAPSTONE_EOF
fi

# ------------------------------------------------------------
# 13. Add README files
# ------------------------------------------------------------

cat > learning/README.md <<'LEARNING_EOF'
# Learning

Experiments, course exercises, notebooks and concept demonstrations.

Code here may be exploratory and does not need to follow production structure.

When something becomes reusable by the final capstone, move or refactor it into:

`capstone/src/enterprise_ai/`
LEARNING_EOF

cat > mini-projects/README.md <<'MP_EOF'
# Mini Projects

Course mini-project submissions and associated materials.

Each project should have its own directory:

- mp01-prompt-lab
- mp02-...
- mp03-...
MP_EOF

# ------------------------------------------------------------
# 14. Show resulting structure
# ------------------------------------------------------------

echo
echo "========================================"
echo " Restructure complete"
echo "========================================"

echo
echo "Git status:"
git status --short

echo
echo "Top-level directories:"
find . -maxdepth 2 \
  -not -path "./.git*" \
  -print | sort

echo
echo "Safety backup branch:"
echo "  $BACKUP_BRANCH"

echo
echo "Nothing has been committed yet."
echo
echo "Review using:"
echo "  git status"
echo "  git diff --stat"
echo "  git diff"
echo
echo "If satisfied:"
echo '  git add -A'
echo '  git commit -m "refactor: restructure repository for learning and capstone"'
echo '  git push origin main'
echo
echo "If something looks wrong:"
echo "  git reset --hard"
echo "or switch to:"
echo "  git switch $BACKUP_BRANCH"
