#!/usr/bin/env python3
"""
scripts/cron_job.py
--------------------
Automated daily prediction pipeline.
Run with a cron schedule or as a standalone script.

Crontab entry (runs every Monday 6 AM):
    0 6 * * 1  cd /path/to/project && python scripts/cron_job.py

Or with APScheduler (included below for Docker / Render deployments):
    python scripts/cron_job.py --scheduler
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def run_pipeline_step(script: str, extra_args: list[str] | None = None) -> bool:
    """Run a Python script as subprocess and log result."""
    cmd = [sys.executable, script] + (extra_args or [])
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error("Step FAILED: %s (exit code %d)", script, result.returncode)
        return False
    logger.info("Step OK: %s", script)
    return True


def run_daily_pipeline() -> None:
    """Execute the full prediction pipeline."""
    run_dt = datetime.now().strftime("%Y-%m-%d %H:%M")
    logger.info("=" * 60)
    logger.info("Retail Demand Forecasting — Daily Pipeline [%s]", run_dt)
    logger.info("=" * 60)

    steps = [
        ("scripts/predict_pipeline.py", None),
    ]

    success = True
    for script, args in steps:
        if not run_pipeline_step(script, args):
            success = False
            break

    if success:
        logger.info("✅  Daily pipeline completed successfully at %s", run_dt)
    else:
        logger.error("❌  Pipeline FAILED at %s", run_dt)
        sys.exit(1)


def run_weekly_retrain() -> None:
    """Retrain models (run weekly on Sunday night)."""
    logger.info("=" * 60)
    logger.info("Weekly Model Retraining")
    logger.info("=" * 60)
    run_pipeline_step("scripts/train_pipeline.py")


def scheduler_mode() -> None:
    """Run via APScheduler — useful in containerised deployments."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        logger.error("Install apscheduler: pip install apscheduler")
        sys.exit(1)

    sched = BlockingScheduler(timezone="UTC")

    # Predict every Monday at 06:00
    sched.add_job(run_daily_pipeline, "cron",
                  day_of_week="mon", hour=6, minute=0,
                  id="daily_predict")

    # Retrain every Sunday at 02:00
    sched.add_job(run_weekly_retrain, "cron",
                  day_of_week="sun", hour=2, minute=0,
                  id="weekly_retrain")

    logger.info("Scheduler started. Jobs:")
    for job in sched.get_jobs():
        logger.info("  %s → next run: %s", job.id, job.next_run_time)

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


def parse_args():
    p = argparse.ArgumentParser(description="Automated pipeline runner")
    p.add_argument("--scheduler", action="store_true",
                   help="Run in persistent APScheduler mode")
    p.add_argument("--retrain",   action="store_true",
                   help="Force a model retraining run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.scheduler:
        scheduler_mode()
    elif args.retrain:
        run_weekly_retrain()
    else:
        run_daily_pipeline()
