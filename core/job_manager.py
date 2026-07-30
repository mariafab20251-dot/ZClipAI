import sqlite3
import json
import uuid
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from .models import Job, JobStatus, Clip
from utils.logging import get_logger

logger = get_logger("job_manager")


class JobManager:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    status TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)
            """)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def create_job(
        self,
        input_video: Path,
        output_dir: Path,
        num_clips: int,
        clip_duration: int,
        config: Dict[str, Any]
    ) -> Job:
        job = Job(
            id=str(uuid.uuid4())[:8],
            input_video=input_video,
            output_dir=output_dir,
            num_clips=num_clips,
            clip_duration=clip_duration,
            config=config,
            created_at=time.time(),
            updated_at=time.time()
        )
        self.save_job(job)
        logger.info("Job created", job_id=job.id, video=str(input_video))
        return job

    def save_job(self, job: Job):
        job.updated_at = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO jobs (id, data, created_at, updated_at, status) VALUES (?, ?, ?, ?, ?)",
                (job.id, json.dumps(job.to_dict()), job.created_at, job.updated_at, job.status.value)
            )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT data FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row:
                return Job.from_dict(json.loads(row["data"]))
        return None

    def get_jobs(self, status: Optional[JobStatus] = None, limit: int = 100) -> List[Job]:
        with self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT data FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status.value, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data FROM jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [Job.from_dict(json.loads(row["data"])) for row in rows]

    def update_job_status(self, job_id: str, status: JobStatus, progress: float = None, step: str = None, error: str = None):
        job = self.get_job(job_id)
        if not job:
            logger.warning("Job not found for status update", job_id=job_id)
            return

        job.status = status
        if progress is not None:
            job.progress = progress
        if step is not None:
            job.current_step = step
        if error is not None:
            job.error = error

        self.save_job(job)
        logger.info("Job status updated", job_id=job_id, status=status.value, progress=progress)

    def add_clip(self, job_id: str, clip: Clip):
        job = self.get_job(job_id)
        if not job:
            return

        clip.id = len(job.clips) + 1
        job.clips.append(clip)
        self.save_job(job)

    def delete_job(self, job_id: str):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
        logger.info("Job deleted", job_id=job_id)

    def cleanup_old_jobs(self, days: int = 30):
        cutoff = time.time() - (days * 86400)
        with self._get_connection() as conn:
            conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM jobs").fetchone()["c"]
            by_status = {}
            for status in JobStatus:
                count = conn.execute("SELECT COUNT(*) as c FROM jobs WHERE status = ?", (status.value,)).fetchone()["c"]
                by_status[status.value] = count
            return {
                "total_jobs": total,
                "by_status": by_status
            }