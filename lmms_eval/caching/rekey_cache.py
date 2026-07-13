"""Re-key existing response-cache entries to a new eval_version.

Cache keys embed ``eval_version`` (by default the lmms-eval git commit hash),
so committing to a dev checkout orphans every cached response.  This tool
recomputes keys under a new version string and inserts the copies into the
consolidated ``cache.db``, so existing responses stay usable after a commit
or after pinning ``LMMS_CACHE_EVAL_VERSION``.

Key components not stored in the DB rows (content_hash, task_fingerprint,
model_fingerprint_hash) are recovered from the JSONL audit logs, which record
them per entry.  Rows without an audit record are skipped and reported.

Idempotent: re-running only inserts keys that don't exist yet, so it is safe
to run again after more shards land (e.g. from a run that was still going).

Usage::

    python -m lmms_eval.caching.rekey_cache /path/to/cache_dir cache-v1
"""

import json
import os
import sqlite3
import sys
from glob import glob

from lmms_eval.caching.response_cache import _SCHEMA_SQL, compute_cache_key


def _load_audit_records(cache_dir: str) -> dict:
    """Map old cache_key -> audit record, from root + per-run audit logs."""
    audit_files = sorted(glob(os.path.join(cache_dir, "cache.audit.jsonl")))
    audit_files += sorted(glob(os.path.join(cache_dir, "runs", "*", "rank_*.audit.jsonl")))
    records = {}
    for path in audit_files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = rec.get("cache_key")
                if key and rec.get("deterministic", True):
                    records[key] = rec
    return records


def rekey_cache_dir(cache_dir: str, new_eval_version: str) -> dict:
    target_db_path = os.path.join(cache_dir, "cache.db")
    records = _load_audit_records(cache_dir)

    db_files = sorted(glob(os.path.join(cache_dir, "cache.db")))
    db_files += sorted(glob(os.path.join(cache_dir, "runs", "*", "rank_*.db")))

    out = sqlite3.connect(target_db_path, timeout=30)
    out.execute("PRAGMA journal_mode=DELETE")
    out.execute("PRAGMA synchronous=FULL")
    out.executescript(_SCHEMA_SQL)

    stats = {"seen": 0, "inserted": 0, "already_current": 0, "no_audit_record": 0}
    seen_old_keys = set()
    for db_path in db_files:
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            continue
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        rows = src.execute("SELECT cache_key, request_type, task_name, doc_id, idx, gen_kwargs, response, created_at FROM responses").fetchall()
        src.close()
        for row in rows:
            old_key = row[0]
            if old_key in seen_old_keys:
                continue
            seen_old_keys.add(old_key)
            stats["seen"] += 1
            rec = records.get(old_key)
            if rec is None:
                stats["no_audit_record"] += 1
                continue
            try:
                gen_kwargs = json.loads(rec.get("gen_kwargs") or "{}")
            except json.JSONDecodeError:
                gen_kwargs = {}
            new_key = compute_cache_key(
                request_type=rec["request_type"],
                task_name=rec["task_name"],
                doc_id=rec["doc_id"],
                gen_kwargs=gen_kwargs,
                idx=rec.get("idx", 0),
                content_hash=rec.get("content_hash", ""),
                task_fingerprint=rec.get("task_fingerprint", ""),
                model_fingerprint_hash=rec.get("model_fingerprint_hash", ""),
                eval_version=new_eval_version,
            )
            if new_key == old_key:
                stats["already_current"] += 1
                continue
            cur = out.execute(
                "INSERT OR IGNORE INTO responses (cache_key, request_type, task_name, doc_id, idx, gen_kwargs, response, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (new_key, *row[1:]),
            )
            stats["inserted"] += cur.rowcount
    out.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("eval_version", new_eval_version))
    out.commit()
    out.close()
    return stats


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    cache_dir, new_eval_version = sys.argv[1], sys.argv[2]
    if not os.path.isdir(cache_dir):
        print(f"error: no such directory: {cache_dir}")
        return 1
    stats = rekey_cache_dir(cache_dir, new_eval_version)
    print(f"{cache_dir}: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
