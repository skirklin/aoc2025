#!/bin/bash
# Deploy static site to GitHub Pages

set -e

echo "=== Generating static site ==="
python -m aoc2025 freeze

echo ""
echo "=== Deploying to gh-pages ==="

# Save current branch
CURRENT_BRANCH=$(git branch --show-current)

# Copy dist to temp location
rm -rf /tmp/aoc-deploy
cp -r dist /tmp/aoc-deploy

# Switch to gh-pages (create if doesn't exist)
if git show-ref --verify --quiet refs/heads/gh-pages; then
    git checkout gh-pages
    # Clean everything except .git
    find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} +
else
    git checkout --orphan gh-pages
    git rm -rf . 2>/dev/null || true
fi

# Copy static files
cp -r /tmp/aoc-deploy/* .

# Add CNAME if configured
if [ -n "$CNAME" ]; then
    echo "$CNAME" > CNAME
fi

# Commit and push
git add .
git commit -m "Deploy static site $(date +%Y-%m-%d)" --allow-empty
git push -f origin gh-pages

# Switch back
git checkout "$CURRENT_BRANCH"

echo ""
echo "=== Done! ==="
echo "Site deployed to gh-pages branch"
echo "Visit: https://$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/' | sed 's/\//.github.io\//')"
