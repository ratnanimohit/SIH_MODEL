#!/usr/bin/env python3
"""
anomaly_detector.py

Two-stage anomaly detection + backup-sensor failover, used by
AssetFeatureBuilder.add() in score_stream.py.

STAGE 1 (hard rule): if any single sensor's z-score has |z| > HARD_RULE_Z_THRESH,
that's an obvious spike -- flagged immediately, no history needed.

STAGE 2 (Isolation Forest): only checked if stage 1 didn't already catch it.
Once an asset has MIN_HISTORY_FOR_IFOREST *clean* ticks of history, an
IsolationForest is trained on that asset's own rolling window of
[z_mean, z_std, z_max, z_min] and re-trained every IFOREST_RETRAIN_EVERY ticks.

A tick is anomalous if either stage flags it. Each asset keeps a rolling
window of the last ANOMALY_WINDOW flags; once ANOMALY_THRESHOLD of them are
anomalous, the asset flips to SENSOR_FAILURE and fails over to a backup
reading (median of the last few known-good z-vectors).

NOTE ON TUNING: the values below are deliberately more conservative than a
naive first pass (higher hard-rule threshold, much longer Isolation-Forest
warm-up, fixed low contamination instead of 'auto'). A tiny/near-constant
training window with contamination='auto' overfits and starts flagging
completely normal ticks as outliers -- that's what was making every asset
in the fleet latch into SENSOR_FAILURE after a short time. These settings
still catch a real spike (see test_anomaly_failover.py, which injects a
20-sigma spike) while giving normal degradation drift a lot more room
before tripping.

RECOVERY: by design, SENSOR_FAILURE does NOT clear itself (a real fault
shouldn't self-heal) -- only mark_maintenance_done() (called by
POST /admin/maintenance/{asset_id}) resets it to NORMAL/primary. This is
intentional and is asserted by test_anomaly_failover.py step 3.
"""

from collections import deque

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
except ImportError:  # pragma: no cover
    IsolationForest = None

# --- stage 1: hard rule ---
HARD_RULE_Z_THRESH = 6.0          # was 4.5 in the original spec; loosened so
                                   # normal end-of-range drift doesn't trip it

# --- stage 2: isolation forest ---
MIN_HISTORY_FOR_IFOREST = 200     # was 20; needs a long warm-up of CLEAN ticks
                                   # before stage 2 turns on at all
IFOREST_WINDOW = 200              # rolling window of clean [z_mean,z_std,z_max,z_min]
                                   # vectors it trains on
IFOREST_RETRAIN_EVERY = 5
IFOREST_CONTAMINATION = 0.02      # fixed & low, instead of 'auto' (which flags
                                   # ~10% of *everything* on a small/near-constant
                                   # window -- the main cause of the runaway failures)

# --- failure / recovery state machine ---
ANOMALY_WINDOW = 10                # look at the last N flags
ANOMALY_THRESHOLD = 3              # >=3 anomalous flags in that window -> SENSOR_FAILURE
BACKUP_MEDIAN_N = 5                # backup vector = median of last N known-good ticks


class AssetHealthMonitor:
    """Per-asset stateful anomaly/failure tracker. One instance lives inside
    each AssetFeatureBuilder (score_stream.py) for the lifetime of that
    asset's stream."""

    def __init__(self):
        self.flag_history = deque(maxlen=ANOMALY_WINDOW)
        self.clean_history = deque(maxlen=IFOREST_WINDOW)  # feeds the IsolationForest
        self.good_zvecs = deque(maxlen=BACKUP_MEDIAN_N)    # feeds the backup median
        self.status = "NORMAL"
        self.active_sensor = "primary"
        self._iforest = None
        self._ticks_since_retrain = 0

    # -- stage checks -------------------------------------------------
    def _hard_rule_hit(self, zs) -> bool:
        return any(abs(z) > HARD_RULE_Z_THRESH for z in zs)

    def _iforest_hit(self, vec):
        """Returns True/False once enough clean history exists, else None
        (stage 2 not active yet)."""
        if IsolationForest is None or len(self.clean_history) < MIN_HISTORY_FOR_IFOREST:
            return None
        if self._iforest is None or self._ticks_since_retrain >= IFOREST_RETRAIN_EVERY:
            X = np.array(self.clean_history, dtype=float)
            self._iforest = IsolationForest(
                n_estimators=100,
                contamination=IFOREST_CONTAMINATION,
                random_state=0,
            ).fit(X)
            self._ticks_since_retrain = 0
        self._ticks_since_retrain += 1
        pred = self._iforest.predict(np.asarray(vec, dtype=float).reshape(1, -1))[0]
        return pred == -1  # sklearn: -1 = outlier, 1 = inlier

    def _backup_vector(self, fallback_vec):
        if not self.good_zvecs:
            return fallback_vec
        return tuple(np.median(np.array(self.good_zvecs, dtype=float), axis=0))

    # -- main entry point ---------------------------------------------
    def update(self, zs):
        """zs: list of this tick's per-sensor z-scores (already
        direction-corrected). Returns (health_dict, backup_vec_or_None)."""
        n = len(zs)
        z_mean = sum(zs) / n
        z_max = max(zs)
        z_min = min(zs)
        z_std = (sum((z - z_mean) ** 2 for z in zs) / n) ** 0.5 if n > 1 else 0.0
        vec = (z_mean, z_std, z_max, z_min)

        hard_hit = self._hard_rule_hit(zs)
        iforest_hit = None if hard_hit else self._iforest_hit(vec)
        is_anomaly = bool(hard_hit or iforest_hit)

        self.flag_history.append(is_anomaly)
        if not is_anomaly:
            self.clean_history.append(vec)
            self.good_zvecs.append(vec)

        anomaly_count = sum(self.flag_history)

        backup_vec = None
        if self.status == "SENSOR_FAILURE":
            # Already failed -- no auto-recovery, keep riding the backup.
            self.active_sensor = "backup"
            backup_vec = self._backup_vector(vec)
        elif anomaly_count >= ANOMALY_THRESHOLD:
            self.status = "SENSOR_FAILURE"
            self.active_sensor = "backup"
            backup_vec = self._backup_vector(vec)

        maintenance_required = self.status == "SENSOR_FAILURE"
        status_message = (
            "SENSOR_FAILURE: reading is unreliable, running on backup "
            "(median of recent known-good readings). Maintenance required."
            if maintenance_required else
            "Sensor operating normally (primary)."
        )

        health = {
            "sensor_status": self.status,
            "active_sensor": self.active_sensor,
            "anomaly_count": anomaly_count,
            "is_anomaly": is_anomaly,
            "hard_rule_hit": hard_hit,
            "iforest_hit": iforest_hit,
            "maintenance_required": maintenance_required,
            "status_message": status_message,
        }
        return health, backup_vec

    def mark_maintenance_done(self):
        """Technician replaced/reset the sensor: wipe all history and go
        back to a clean NORMAL/primary state."""
        self.status = "NORMAL"
        self.active_sensor = "primary"
        self.flag_history.clear()
        self.clean_history.clear()
        self.good_zvecs.clear()
        self._iforest = None
        self._ticks_since_retrain = 0
