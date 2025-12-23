# AoC 2025 Benchmark Harness

A benchmarking harness for comparing AI models (Claude Haiku, Sonnet, Opus) on Advent of Code 2025 problems.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a `.env` file with your AoC session cookie:
```
AOC_SESSION=your_session_cookie_here
```

## Usage

### Fetch Problems
```bash
python -m aoc2025 fetch <day>
```

### Run a Solution
```bash
python -m aoc2025 run <day>
```

### Benchmark a Model
```bash
python -m aoc2025 benchmark <day> --model haiku
python -m aoc2025 benchmark <day> --model sonnet
python -m aoc2025 benchmark <day> --model opus
```

### Run All Benchmarks
```bash
./run_benchmarks.sh                    # All models, all days
./run_benchmarks.sh haiku              # Just haiku
./run_benchmarks.sh "haiku sonnet"     # Multiple models
./run_benchmarks.sh haiku "1 2 3"      # Specific days
```

### View Dashboard Locally
```bash
python -m aoc2025 dashboard
```

## Deployment

The dashboard can be exported as a static site and hosted on GitHub Pages.

### 1. Generate Static Site
```bash
python -m aoc2025 freeze
```

This creates a `dist/` folder with static HTML files.

### 2. Preview Locally
```bash
cd dist && python -m http.server 8000
# Visit http://localhost:8000
```

### 3. Deploy to GitHub Pages

**Option A: Manual deploy**
```bash
# From main branch, generate the static site
python -m aoc2025 freeze

# Create/update gh-pages branch
git checkout --orphan gh-pages-new
git rm -rf .
cp -r dist/* .
git add .
git commit -m "Deploy static site"
git branch -D gh-pages 2>/dev/null
git branch -m gh-pages
git push -f origin gh-pages
git checkout main
```

**Option B: Using the deploy script**
```bash
./deploy.sh
```

### 4. Configure GitHub Pages

1. Go to your repo on GitHub
2. Settings → Pages
3. Source: Deploy from a branch
4. Branch: `gh-pages` / `/ (root)`
5. Save

Your site will be available at `https://<username>.github.io/aoc2025/`

### 5. Custom Domain (Optional)

To use a custom domain like `aoc.kirkl.in`:

1. In your DNS provider, add a CNAME record:
   - Host: `aoc`
   - Value: `<username>.github.io`

2. In GitHub repo Settings → Pages → Custom domain:
   - Enter: `aoc.yourdomain.com`
   - Check "Enforce HTTPS"

3. Add a `CNAME` file to gh-pages branch:
   ```bash
   echo "aoc.yourdomain.com" > CNAME
   git add CNAME && git commit -m "Add CNAME" && git push
   ```

## Project Structure

```
aoc2025/
├── aoc2025/
│   ├── cli.py          # Command-line interface
│   ├── core.py         # Core utilities (fetch, submit, etc.)
│   ├── dashboard.py    # Flask web dashboard
│   ├── db.py           # SQLite database for benchmark results
│   ├── freeze.py       # Static site generator
│   ├── days/           # Solution files (day01.py, day02.py, ...)
│   └── templates/      # Jinja2 templates for dashboard
├── .cache/             # Problem data, inputs, benchmark runs (gitignored)
├── dist/               # Generated static site (gitignored)
└── run_benchmarks.sh   # Script to run all benchmarks
```

## Data Storage

Benchmark data is stored in `.cache/2025/`:
- `aoc.db` - SQLite database with run results
- `runs/<run_id>/` - Individual run artifacts (solution code, Claude output)
- `<day>/problem.md` - Problem statements
- `<day>/inputs/` - Input files
