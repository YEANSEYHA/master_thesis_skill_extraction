import asyncio
import subprocess
import os
from crawler import AutonomousCrawler
from crawler.storage import upload_run_to_r2
from crawler.db import create_crawl_history_table
from config import MODE
from datetime import datetime
import json


def ensure_flaresolverr():
    """Start FlareSolverr Docker container if not already running."""
    try:
        # Check if container exists and is running
        result = subprocess.run(
            ["sudo", "docker", "inspect", "-f", "{{.State.Running}}", "flaresolverr"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip() == "true":
            print("✅ FlareSolverr already running")
            return True

        # Container exists but stopped — start it
        if result.returncode == 0:
            print("🔄 Starting existing FlareSolverr container...")
            subprocess.run(["sudo", "docker", "start", "flaresolverr"], check=True)
            print("✅ FlareSolverr started")
            return True

        # Container doesn't exist — create and run it
        print("🚀 Launching FlareSolverr container...")
        subprocess.run([
            "sudo", "docker", "run", "-d",
            "--name", "flaresolverr",
            "-p", "8191:8191",
            "ghcr.io/flaresolverr/flaresolverr:latest"
        ], check=True)
        print("✅ FlareSolverr launched")
        return True

    except FileNotFoundError:
        print("⚠️  Docker not found — FlareSolverr will not be available")
        return False
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to start FlareSolverr: {e}")
        return False

class CrawlTracker:
    """Track crawl results across multiple sites"""
    
    def __init__(self):
        self.results = {
            "success": [],
            "blocked": [],
            "failed": [],
            "pagination_found": []
        }
        self.start_time = None
        self.end_time = None
    
    def add_success(self, url: str, pages_saved: int, llm_used: int):
        self.results["success"].append({
            "url": url,
            "pages_saved": pages_saved,
            "llm_used": llm_used,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_blocked(self, url: str, reason: str = "Unknown"):
        self.results["blocked"].append({
            "url": url,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_failed(self, url: str, error: str):
        self.results["failed"].append({
            "url": url,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })
    
    def add_pagination(self, url: str):
        self.results["pagination_found"].append(url)
    
    def print_summary(self):
        """Print detailed summary"""
        total = len(self.results["success"]) + len(self.results["blocked"]) + len(self.results["failed"])

        print(f"\n{'='*80}")
        print("📊 CRAWL SUMMARY")
        print(f"{'='*80}")
        print(f"Total Sites Attempted: {total}")
        if total > 0:
            print(f"✅ Success: {len(self.results['success'])} ({len(self.results['success'])/total*100:.1f}%)")
            print(f"❌ Blocked: {len(self.results['blocked'])} ({len(self.results['blocked'])/total*100:.1f}%)")
            print(f"💥 Failed:  {len(self.results['failed'])} ({len(self.results['failed'])/total*100:.1f}%)")
        else:
            print("No sites were attempted.")
        print(f"🔢 Pagination Found: {len(self.results['pagination_found'])}")
        
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            print(f"⏱️  Total Time: {duration:.1f}s ({duration/60:.1f} min)")
        
        print(f"{'='*80}\n")
        
        # Success details
        if self.results["success"]:
            print("✅ SUCCESSFUL SITES:")
            total_pages = 0
            total_llm = 0
            for item in self.results["success"]:
                pages = item.get("pages_saved", 0)
                llm = item.get("llm_used", 0)
                total_pages += pages
                total_llm += llm
                print(f"  ✓ {item['url']}")
                print(f"    └─ Pages: {pages}, LLM calls: {llm}")
            print(f"  📄 Total pages saved: {total_pages}")
            print(f"  🤖 Total LLM calls: {total_llm}\n")
        
        # Blocked details
        if self.results["blocked"]:
            print("❌ BLOCKED SITES:")
            for item in self.results["blocked"]:
                print(f"  ✗ {item['url']}")
                print(f"    └─ Reason: {item.get('reason', 'Unknown')}")
            print()
        
        # Failed details
        if self.results["failed"]:
            print("💥 FAILED SITES:")
            for item in self.results["failed"]:
                print(f"  ✗ {item['url']}")
                print(f"    └─ Error: {item.get('error', 'Unknown')}")
            print()
        
        # Pagination details
        if self.results["pagination_found"]:
            print("🔢 SITES WITH PAGINATION:")
            for url in self.results["pagination_found"]:
                print(f"  • {url}")
            print()
    
    def save_report(self, filename: str = "crawl_report.json"):
        """Save detailed JSON report"""
        report = {
            "summary": {
                "total": len(self.results["success"]) + len(self.results["blocked"]) + len(self.results["failed"]),
                "success": len(self.results["success"]),
                "blocked": len(self.results["blocked"]),
                "failed": len(self.results["failed"]),
                "pagination_found": len(self.results["pagination_found"]),
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None
            },
            "results": self.results
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Report saved to: {filename}")


async def crawl_single_site(start_url: str, max_pages: int = 1, tracker: CrawlTracker = None, run_id: str = ""):
    """Crawl a single website with tracking"""
    print(f"\n{'='*80}")
    print(f"Starting crawl for: {start_url}")
    print(f"Max pages: {max_pages}")
    print(f"{'='*80}\n")

    try:
        crawler = AutonomousCrawler(
            start_url=start_url,
            max_pages=max_pages,
            run_id=run_id
        )
        
        # Add progress monitoring
        crawl_task = asyncio.create_task(crawler.crawl())
        
        # Monitor progress while crawling
        last_saved = 0
        while not crawl_task.done():
            await asyncio.sleep(2)  # Check every 2 seconds
            current_saved = crawler.saved
            if current_saved > last_saved:
                print(f"📄 Progress: {current_saved}/{max_pages} pages saved")
                last_saved = current_saved
        
        # Wait for completion
        await crawl_task
        
        # Final results...
        if crawler.saved > 0:
            if tracker:
                tracker.add_success(
                    url=start_url,
                    pages_saved=crawler.saved,
                    llm_used=crawler.metadata.get("llm_used", 0)
                )
                
                if crawler.metadata.get("llm_used", 0) > 0:
                    tracker.add_pagination(start_url)
            
            print(f"\n✅ Success: {crawler.saved} pages saved")
        else:
            if crawler.metadata.get("blocked"):
                if tracker:
                    tracker.add_blocked(start_url, "Anti-bot protection detected")
                print(f"❌ Blocked: No pages saved")
            else:
                if tracker:
                    tracker.add_failed(start_url, "No content extracted")
                print(f"⚠️  Warning: No pages saved")
    
    except Exception as e:
        if tracker:
            tracker.add_failed(start_url, str(e))
        print(f"💥 Failed: {e}")
    
    print(f"\n{'='*80}")
    print(f"Completed: {start_url}")
    print(f"{'='*80}\n")

async def crawl_multiple_sites(urls: list, max_pages: int = 1, concurrency: int = 3):
    """Crawl multiple websites concurrently with tracking.

    Sites are independent servers, so crawling `concurrency` of them at once
    multiplies throughput while each individual site still gets the polite
    1s-per-request delay from its own crawler."""

    # Initialize tracker
    tracker = CrawlTracker()
    tracker.start_time = datetime.now()

    # Generate run ID (timestamp folder)
    run_id = tracker.start_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Ensure crawl_history table exists
    create_crawl_history_table()

    print(f"\n{'🚀'*40}")
    print(f"STARTING MULTI-SITE CRAWL")
    print(f"Run ID: {run_id}")
    print(f"Total websites to crawl: {len(urls)} ({concurrency} at a time)")
    print(f"Started at: {tracker.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'🚀'*40}\n")

    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def crawl_with_limit(index: int, url: str):
        nonlocal done
        async with sem:
            # Stagger startups so browsers/DB don't spike at once
            await asyncio.sleep((index - 1) % concurrency * 3)
            await crawl_single_site(url, max_pages, tracker, run_id)
            done += 1
            print(f"\n📈 Sites finished: {done}/{len(urls)} | "
                  f"✅ {len(tracker.results['success'])} | "
                  f"❌ {len(tracker.results['blocked'])} | "
                  f"💥 {len(tracker.results['failed'])}")

    await asyncio.gather(*(crawl_with_limit(i, u) for i, u in enumerate(urls, 1)))

    tracker.end_time = datetime.now()
    
    # Print final summary
    tracker.print_summary()
    
    # Save report and lists into the run folder
    run_dir = f"outputs/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    tracker.save_report(f"{run_dir}/crawl_report.json")
    save_simple_lists(tracker, run_dir)

    # Upload to Cloudflare R2 (production only)
    if MODE == "production":
        upload_run_to_r2(run_id)
    else:
        print(f"[R2] Skipping upload (MODE={MODE})")


def save_simple_lists(tracker: CrawlTracker, run_dir: str):
    """Save simple text lists of URLs into the run folder"""

    # Blocked sites
    if tracker.results["blocked"]:
        with open(f"{run_dir}/blocked_sites.txt", "w") as f:
            f.write("# Sites blocked by anti-bot protection\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            for item in tracker.results["blocked"]:
                f.write(f"{item['url']}\n")
        print(f"📝 Blocked sites saved to: {run_dir}/blocked_sites.txt")

    # Failed sites
    if tracker.results["failed"]:
        with open(f"{run_dir}/failed_sites.txt", "w") as f:
            f.write("# Sites that failed to crawl\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            for item in tracker.results["failed"]:
                f.write(f"{item['url']} - {item['error']}\n")
        print(f"📝 Failed sites saved to: {run_dir}/failed_sites.txt")

    # Success sites
    if tracker.results["success"]:
        with open(f"{run_dir}/success_sites.txt", "w") as f:
            f.write("# Successfully crawled sites\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            for item in tracker.results["success"]:
                f.write(f"{item['url']} - {item['pages_saved']} pages\n")
        print(f"📝 Success sites saved to: {run_dir}/success_sites.txt")


async def main():
    # FlareSolverr is the last-resort renderer for Cloudflare-challenged
    # pages — warn but keep going without it (HTTP/stealth handle most sites)
    if not ensure_flaresolverr():
        print("⚠️  Continuing WITHOUT FlareSolverr — Cloudflare-protected "
              "pages may be skipped. Start Docker + the container to enable it.")

    # Multiple sites crawl
    urls = [
        "https://www.hrincjobs.com/",
        "https://jobify.works/",
        "https://workingna.com/",
        "https://www.jobnet.com.kh/",
        "https://camupjob.com/",
        "https://bongthom.com/",
        "https://www.cambojob.com/",
        "https://www.khmerjob.com/",
        "https://thetalent4u.com/",
        "https://www.camhr.com/",
    ]
    
    await crawl_multiple_sites(
        urls=urls,
        max_pages=2000,   # Pages per site
        concurrency=3,    # Sites crawled in parallel (each stays polite per-site)
    )

if __name__ == "__main__":
    asyncio.run(main())