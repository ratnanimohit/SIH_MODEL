#!/usr/bin/env python3
"""
anomaly_detector.py

Adds an anomaly-detection + sensor-failure/backup-failover layer on top of
the existing z-score pipeline in score_stream.py. Nothing here touches the
RUL model or MODEL_FEATURE_ORDER -- this is a parallel health signal, not a
new model input, so xgboost_rul_model.joblib does NOT need retraining.

TWO-STAGE DETECTION (per tick, per asset)
------------------------------------------
STAGE 1 -- hard rule check (cheap, catches obvious spikes first):
  Reuses the SAME per-sensor z-scores AssetFeatureBuilder already computes
  (direction-corrected (value - baseline_mean) / baseline_std). If any single
  sensor's |z| exceeds HARD_Z_THRESHOLD, that tick is flagged immediately --
  no history needed, no model, just a threshold.

STAGE 2 -- Isolation Forest (catches subtler anomalies the hard rule misses,
  e.g. a combination of sensors that's individually unremarkable but jointly
  unusual):
  Once an asset has IFOREST_MIN_SAMPLES ticks of *clean* history, an
  IsolationForest is trained on that asset's own rolling window of
  [z_mean, z_std, z_max, z_min] vectors and used to score the current tick.
  scikit-learn's IsolationForest has no partial_fit/incremental mode, so
  "retrain every few ticks on a rolling window" (done here every
  IFOREST_REFIT_EVERY ticks) is the standard workaround for streaming data.
  Stage 2 only runs after Stage 1 -- if the hard rule already caught it,
  there's no need to also ask the forest.

A tick counts as anomalous if EITHER stage flags it.

SENSOR FAILURE + BACKUP FAILOVER
---------------------------------
Each asset keeps a sliding window of the last ANOMALY_WINDOW flags. Once
ANOMALY_FAILURE_COUNT (default 3) of them are anomalous, the asset flips to
SENSOR_FAILURE, fails over to a backup reading, and is marked
maintenance_required=True. It STAYS in SENSOR_FAILURE -- this is meant to
model a real fault needing a technician, not something that should quietly
disappear mid-demo -- until either:
  (a) someone calls the manual "maintenance done" action (wired to
      POST /admin/maintenance/{asset_id} in app.py, simulating a technician
      resetting/replacing the sensor), or
  (b) ANOMALY_AUTO_RECOVER=true is set, in which case RECOVERY_CLEAN_STREAK
      consecutive clean ticks will also clear it automatically (off by
      default, see ENV VARS below).

IMPORTANT ASSUMPTION: this fleet/schema has no second physical sensor per
measurement (see asset_fleet.py's SENSOR_TEMPLATES -- each component has
2-3 *different* sensors, not redundant pairs of the same one). So "backup
sensor" here means: stop trusting the live (possibly glitching) reading and
fall back to the median of the last few known-good (non-anomalous) readings
for that asset, so the RUL model keeps scoring off a trusted recent trend
instead of a corrupted spike. If you later add real redundant hardware
sensors, swap `AssetHealthMonitor.update()`'s use of `_backup_vector()` for
your actual secondary sensor's z-vector -- everything downstream (the
SENSOR_FAILURE / active_sensor plumbing) stays the same.

ENV VARS (all optional, sane defaults)
  ANOMALY_HARD_Z_THRESHOLD   |z| above this on one sensor = obvious anomaly (default 4.5)
  ANOMALY_WINDOW             ticks considered for the rolling anomaly count (default 10)
  ANOMALY_FAILURE_COUNT      flagged ticks in the window that trigger failure (default 3)
  ANOMALY_AUTO_RECOVER       "true" to let a clean streak auto-clear a failure (default false --
                              off, so a failure sticks until /admin/maintenance is called)
  ANOMALY_RECOVERY_STREAK    consecutive clean ticks needed for auto-recovery, only used
                              if ANOMALY_AUTO_RECOVER=true (default 5)
  IFOREST_MIN_SAMPLES        clean ticks needed before the forest is trusted (default 20)
  IFOREST_REFIT_EVERY        retrain cadence, in ticks (default 5)
  IFOREST_WINDOW             max clean ticks kept per asset for training (default 100)
  IFOREST_CONTAMINATION      expected anomaly fraction passed to IsolationForest (default 0.1)
"""

import math
import os
from collections import deque

import numpy as np
from sklearn.ensemble import IsolationForest

HARD_Z_THRESHOLD = float(os.environ.get("ANOMALY_HARD_Z_THRESHOLD", "4.5"))
ANOMALY_WINDOW = int(os.environ.get("ANOMALY_WINDOW", "10"))
ANOMALY_FAILURE_COUNT = int(os.environ.get("ANOMALY_FAILURE_COUNT", "3"))
AUTO_RECOVER = os.environ.get("ANOMALY_AUTO_RECOVER", "false").strip().lower() == "true"
RECOVERY_CLEAN_STREAK = int(os.environ.get("ANOMALY_RECOVERY_STREAK", "5"))
IFOREST_MIN_SAMPLES = int(os.environ.get("IFOREST_MIN_SAMPLES", "20"))
IFOREST_REFIT_EVERY = int(os.environ.get("IFOREST_REFIT_EVERY", "5"))
IFOREST_WINDOW = int(os.environ.get("IFOREST_WINDOW", "100"))
IFOREST_CONTAMINATION = float(os.environ.get("IFOREST_CONTAMINATION", "0.1"))


def status_message(status: str, active_sensor: str, anomaly_count: int) -> str:
    """Human-readable line for a dashboard -- this is the literal text a
    frontend can show under an asset when it's failed."""
    if status == "SENSOR_FAILURE":
        return (f"Sensor failure detected ({anomaly_count} anomalies in the last "
                f"{ANOMALY_WINDOW} readings). Running on BACKUP sensor -- maintenance required.")
    return "Sensor operating normally (primary)."


class AssetHealthMonitor:
    """One instance per asset (created alongside its AssetFeatureBuilder).
    Call .update(zs) once per tick with the list of per-sensor z-scores for
    that tick. Returns (health: dict, backup_vector: list|None).

    backup_vector, when not None, is [z_mean, z_std, z_max, z_min] computed
    from recent known-good ticks -- the caller (AssetFeatureBuilder) should
    use these in place of the freshly-computed aggregates while
    health['sensor_status'] == 'SENSOR_FAILURE'.
    """

    def __init__(self):
        self.flags = deque(maxlen=ANOMALY_WINDOW)
        self.good_history = deque(maxlen=IFOREST_WINDOW)  # clean-tick [z_mean,z_std,z_max,z_min] vectors
        self.clean_streak = 0
        self.status = "NORMAL"
        self.active_sensor = "primary"
        self._iforest = None
        self._ticks_since_fit = 0

    @staticmethod
    def _aggregate(zs):
        z_mean = sum(zs) / len(zs)
        z_max, z_min = max(zs), min(zs)
        z_std = math.sqrt(sum((z - z_mean) ** 2 for z in zs) / len(zs)) if len(zs) > 1 else 0.0
        return [z_mean, z_std, z_max, z_min]

    @staticmethod
    def _hard_rule_hit(zs):
        """STAGE 1: simple, explainable threshold check -- any one sensor
        reading whose z-score is an obvious outlier."""
        return any(abs(z) > HARD_Z_THRESHOLD for z in zs)

    def _iforest_hit(self, vec):
        """STAGE 2: only reached if Stage 1 didn't already flag the tick.
        Returns None (not yet trusted) until enough clean history exists."""
        if len(self.good_history) < IFOREST_MIN_SAMPLES:
            return None
        if self._iforest is None or self._ticks_since_fit >= IFOREST_REFIT_EVERY:
            X = np.array(self.good_history)
            self._iforest = IsolationForest(
                n_estimators=100,
                contamination=IFOREST_CONTAMINATION,
                random_state=42,
            ).fit(X)
            self._ticks_since_fit = 0
        self._ticks_since_fit += 1
        pred = self._iforest.predict([vec])[0]  # -1 = anomaly, 1 = normal
        return pred == -1

    def _backup_vector(self):
        """Median of recent known-good aggregate vectors -- see module
        docstring for why this stands in for a real redundant sensor."""
        if not self.good_history:
            return None
        return np.median(np.array(self.good_history), axis=0).tolist()

    def update(self, zs):
        vec = self._aggregate(zs)

        hard_hit = self._hard_rule_hit(zs)
        iforest_hit = False if hard_hit else self._iforest_hit(vec)  # skip stage 2 if stage 1 already caught it
        is_anomaly = hard_hit or bool(iforest_hit)

        self.flags.append(is_anomaly)
        anomaly_count = sum(self.flags)

        if is_anomaly:
            self.clean_streak = 0
        else:
            self.clean_streak += 1
            self.good_history.append(vec)  # only train/backup off trusted, clean ticks

        if self.status == "SENSOR_FAILURE":
            # While failed, recovery is judged purely on the clean streak --
            # NOT the rolling window count, which can still hold the old
            # anomalies that caused the failure and would otherwise re-trigger
            # it the instant we recovered. Off by default (see AUTO_RECOVER) --
            # a real sensor failure shouldn't silently clear itself; use
            # mark_maintenance_done() / POST /admin/maintenance/{asset_id}.
            if AUTO_RECOVER and self.clean_streak >= RECOVERY_CLEAN_STREAK:
                self.status = "NORMAL"
                self.active_sensor = "primary"
                self.flags.clear()
                anomaly_count = 0
        elif anomaly_count >= ANOMALY_FAILURE_COUNT:
            self.status = "SENSOR_FAILURE"
            self.active_sensor = "backup"

        health = {
            "sensor_status": self.status,
            "active_sensor": self.active_sensor,
            "anomaly_count": anomaly_count,      # flagged ticks within the last ANOMALY_WINDOW
            "is_anomaly": is_anomaly,
            "hard_rule_hit": hard_hit,
            "iforest_hit": iforest_hit,          # None = forest not warmed up yet, else True/False
            "maintenance_required": self.status == "SENSOR_FAILURE",
            "status_message": status_message(self.status, self.active_sensor, anomaly_count),
        }
        backup_vector = self._backup_vector() if self.status == "SENSOR_FAILURE" else None
        return health, backup_vector

    def mark_maintenance_done(self):
        """Manually clear a SENSOR_FAILURE, as if a technician just
        replaced/reset the physical sensor. Wired to
        POST /admin/maintenance/{asset_id} in app.py so this is demoable
        live: trigger a failure, show it on the dashboard, call this, show
        it clear again."""
        self.status = "NORMAL"
        self.active_sensor = "primary"
        self.flags.clear()
        self.clean_streak = 0
