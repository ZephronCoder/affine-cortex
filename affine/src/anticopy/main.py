"""
Anti-Copy Detection Service - Main Entry Point

Runs as an independent background service that periodically fetches miners
from metagraph and runs copy detection using logprob/hidden-state signals.
"""

import asyncio
import json
import math
import time
import signal
import click
from collections import defaultdict
from typing import Dict, List
from affine.core.setup import logger, setup_logging, NETUID
from affine.database import init_client, close_client
from affine.utils.subtensor import get_subtensor
from affine.database.dao.anti_copy import AntiCopyDAO
from .detector import AntiCopyDetector
from .loader import LogprobsLoader


DEFAULT_INTERVAL = 21600  # 6 hours


class AntiCopyService:
    """Periodic anti-copy detection service."""

    def __init__(self, interval: int = DEFAULT_INTERVAL):
        self.interval = interval
        self._running = False
        self._task = None

    async def _fetch_miners(self):
        """Fetch active miners from metagraph."""
        subtensor = await get_subtensor()
        meta = await subtensor.metagraph(NETUID)
        commits = await subtensor.get_all_revealed_commitments(NETUID)

        miners = []
        for uid in range(len(meta.hotkeys)):
            hotkey = meta.hotkeys[uid]
            if hotkey not in commits:
                continue
            try:
                block, commit_data = commits[hotkey][-1]
                data = json.loads(commit_data)
                revision = data.get("revision", "")
                model = data.get("model", "")
                if hotkey and revision and model:
                    miners.append({
                        "uid": uid,
                        "hotkey": hotkey,
                        "revision": revision,
                        "model": model,
                        "block": int(block) if uid != 0 else 0,
                    })
            except Exception:
                continue

        logger.info(f"[AntiCopy] Fetched {len(miners)} active miners from chain")
        return miners

    def _build_copy_groups(
        self, copy_pairs, miner_info: Dict[int, dict]
    ) -> List[dict]:
        """Build per-model records from copy pairs using union-find.

        For each group of copies, the miner with the earliest block is
        considered the original; the rest are copiers.

        Returns:
            List of dicts ready for DAO.save_round()
        """
        # Union-Find to group connected copy pairs
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for pair in copy_pairs:
            union(pair.uid_a, pair.uid_b)

        # Group UIDs by root
        groups: Dict[int, set] = defaultdict(set)
        all_copy_uids = set()
        for pair in copy_pairs:
            all_copy_uids.update([pair.uid_a, pair.uid_b])
        for uid in all_copy_uids:
            groups[find(uid)].add(uid)

        # Build pair lookup: (uid_a, uid_b) -> CopyPair
        pair_map = {}
        for pair in copy_pairs:
            pair_map[(pair.uid_a, pair.uid_b)] = pair
            pair_map[(pair.uid_b, pair.uid_a)] = pair

        results = []
        for group_uids in groups.values():
            # Original = earliest block in group
            sorted_uids = sorted(
                group_uids, key=lambda u: miner_info[u]["block"]
            )
            original_uid = sorted_uids[0]

            # Only store copiers, skip the original
            for uid in sorted_uids[1:]:
                info = miner_info[uid]
                pair = pair_map.get((uid, original_uid)) or pair_map.get((original_uid, uid))
                orig_info = miner_info[original_uid]

                copy_entry = {
                    "uid": original_uid,
                    "hotkey": orig_info["hotkey"],
                    "model": orig_info["model"],
                }
                if pair:
                    for key, val in [
                        ("logprobs_cosine", pair.cosine_similarity),
                        ("hs_cosine", pair.hs_cosine),
                        ("js_div", pair.js_divergence),
                    ]:
                        if val is not None and not (isinstance(val, float) and math.isnan(val)):
                            copy_entry[key] = val
                    copy_entry["n_tasks"] = pair.n_tasks

                results.append({
                    "uid": uid,
                    "hotkey": info["hotkey"],
                    "model": info["model"],
                    "revision": info["revision"],
                    "block": info["block"],
                    "is_copy": True,
                    "copy_of": [copy_entry],
                })

        return results

    async def _run_detection(self):
        """Run one round of copy detection."""
        miners = await self._fetch_miners()
        if len(miners) < 2:
            logger.warning("[AntiCopy] Not enough miners for comparison, skipping")
            return

        # Build uid -> miner info lookup
        miner_info = {m["uid"]: m for m in miners}

        loader = LogprobsLoader()
        miner_data = await loader.load_all_miners(miners)
        logger.info(f"[AntiCopy] Loaded logprobs for {len(miner_data)}/{len(miners)} miners")

        if len(miner_data) < 2:
            logger.warning("[AntiCopy] Not enough miners with logprob data, skipping")
            return

        detector = AntiCopyDetector()
        results = detector.detect(miner_data)

        copy_pairs = [r for r in results if r.is_copy]
        logger.info(
            f"[AntiCopy] Detection complete: {len(copy_pairs)} copy pairs "
            f"out of {len(results)} pairs evaluated"
        )
        for r in copy_pairs:
            logger.warning(f"[AntiCopy] {r}")

        # Save to DB
        if copy_pairs:
            records = self._build_copy_groups(copy_pairs, miner_info)
            dao = AntiCopyDAO()
            round_ts = int(time.time())
            await dao.save_round(records, round_timestamp=round_ts)
            copy_count = sum(1 for r in records if r["is_copy"])
            logger.info(
                f"[AntiCopy] Saved {len(records)} records "
                f"({copy_count} copies) to DB"
            )

    async def _loop(self):
        """Background detection loop."""
        while self._running:
            try:
                await self._run_detection()
            except Exception as e:
                logger.error(f"[AntiCopy] Error in detection loop: {e}", exc_info=True)
            await asyncio.sleep(self.interval)

    async def start(self):
        """Start background detection."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"[AntiCopy] Service started (interval={self.interval}s)")

    async def stop(self):
        """Stop background detection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[AntiCopy] Service stopped")


async def run_service(interval: int = DEFAULT_INTERVAL):
    """Run the anti-copy detection service."""
    logger.info("Starting Anti-Copy Detection Service")

    try:
        await init_client()
        logger.info("Database client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    shutdown_event = asyncio.Event()

    def handle_shutdown(sig):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))

    service = None
    try:
        service = AntiCopyService(interval=interval)
        await service.start()
        await shutdown_event.wait()
    except Exception as e:
        logger.error(f"Error running AntiCopyService: {e}", exc_info=True)
        raise
    finally:
        if service:
            try:
                await service.stop()
            except Exception as e:
                logger.error(f"Error stopping AntiCopyService: {e}")
        try:
            await close_client()
            logger.info("Database client closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")

    logger.info("Anti-Copy Detection Service shut down successfully")


@click.command()
@click.option(
    "-v", "--verbosity",
    default=None,
    type=click.Choice(["0", "1", "2", "3"]),
    help="Logging verbosity: 0=CRITICAL, 1=INFO, 2=DEBUG, 3=TRACE"
)
@click.option(
    "--interval",
    default=DEFAULT_INTERVAL,
    type=int,
    help=f"Detection interval in seconds (default: {DEFAULT_INTERVAL})"
)
def main(verbosity, interval):
    """
    Affine Anti-Copy Detection Service.

    Periodically checks all miners for model copying using
    logprob cosine and hidden-state signals.
    """
    if verbosity is not None:
        setup_logging(int(verbosity))
    asyncio.run(run_service(interval=interval))


if __name__ == "__main__":
    main()
