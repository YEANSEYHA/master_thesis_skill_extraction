# Autonomous Crawl Agent

Autonomous web crawler for Cambodian job websites. Handles anti-bot protections (Cloudflare) using a 3-tier rendering fallback, and uses LLM-assisted navigation to discover pagination patterns.

## Project Structure

```
├── main.py                     # Entry point
├── config.py                   # Configuration
├── crawler/                    # Crawler package
│   ├── core.py                 # BFS crawler with deduplication
│   ├── renderer.py             # Playwright → Undetected → FlareSolverr
│   ├── pagination.py           # Pagination detection (BS4 + LLM)
│   ├── navigator.py            # LLM navigation pattern discovery
│   ├── links.py                # Link extraction
│   ├── signals.py              # Content hashing, block detection
│   └── storage.py              # HTML/metadata saving
├── data_collection/            # Collected HTML sample data
├── data_preprocessing/         # HTML classification scripts
└── Results/                    # Crawl output
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally with a model (default: `qwen3:8b`)
- Playwright browsers installed: `playwright install chromium`
- (Optional) [FlareSolverr](https://github.com/FlareSolverr/FlareSolverr) for Cloudflare bypass

## Setup

### Install dependencies

```bash
pip install playwright beautifulsoup4 lxml requests
playwright install chromium
```

### Start Ollama

```bash
ollama serve
ollama pull qwen3:8b
```

### (Optional) Start FlareSolverr

```bash
docker run -d \
  --name flaresolverr \
  -p 8191:8191 \
  --restart unless-stopped \
  ghcr.io/flaresolverr/flaresolverr:latest
```

Test it:
```bash
curl http://localhost:8191
```

## Usage

```bash
python main.py
```

Crawls 10 Cambodian job sites sequentially with a 5-second delay between sites. Results are saved to `Results/Model_<model>/Site_<domain>/`.

### Configuration

Environment variables:
- `ACTIVE_MODEL` - Ollama model name (default: `qwen3:8b`)
- `OLLAMA_URL` - Ollama API endpoint (default: `http://localhost:11434/api/generate`)

### Output

- `Results/` - Crawled HTML pages organized by model and site
- `crawl_report.json` - Detailed crawl report
- `success_sites.txt` - Successfully crawled sites
- `blocked_sites.txt` - Sites blocked by anti-bot protection
- `failed_sites.txt` - Sites that failed to crawl

## How It Works

1. **Rendering** - 3-tier fallback: Playwright stealth → Undetected-Playwright → FlareSolverr
2. **Link extraction** - Extracts same-domain links from each page
3. **Pagination detection** - BeautifulSoup fast check, then LLM fallback
4. **LLM navigation** - Discovers URL patterns like `/page/{n}` for pagination
5. **Deduplication** - SHA256 content hashing prevents saving duplicate pages
6. **Block detection** - Identifies Cloudflare challenges and HTTP error pages
