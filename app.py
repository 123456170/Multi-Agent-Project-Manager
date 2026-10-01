from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
import sqlite3
import tempfile
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

# ============================================================
# Multi-Agent Project Manager - Agentic AI Simulated Prototype
# Full single-file Streamlit application.
# Safe autonomy: all actions are simulator-only.
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Project Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_VERSION = "mpm-agentic-pm/1.0.0"
CONFIG_VERSION = "default/1.0.0"

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "demo_seconds": 120,
    "detection_threshold": 0.58,
    "risk_action_threshold": 45.0,
    "autonomy_mode": "safe_simulation",  # observe | recommend_only | safe_simulation | manual
    "noise_scale": 0.05,
    "live_ticks": 90,
    "live_sleep": 0.20,
    "demo_mode": True,
    "persist_max_ticks": 1200,
}

VIDEO_WIDTH = 960
VIDEO_HEIGHT = 544
VIDEO_FPS = 2  # 120 frames / 2 fps = 60 seconds.


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def str_hash(value: str) -> int:
    return sum(value.encode("utf-8")) % 1000


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def det_id(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def model_to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, BaseModel):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return dict(obj)


def mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def phase_label(t: int) -> str:
    if t < 20:
        return "Normal baseline"
    if t < 45:
        return "Anomaly injection"
    if t < 70:
        return "Persistent detection"
    if t < 90:
        return "AI explanation"
    if t < 105:
        return "Investigation / risk"
    if t < 115:
        return "Policy-gated simulated action"
    return "Verification / evidence"


# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------

class Observation(BaseModel):
    id: str
    t: int
    entity_id: str
    metric: str
    value: float
    quality: float = 1.0
    source: str = "synthetic_world"
    meta: Dict[str, Any] = Field(default_factory=dict)


class Detection(BaseModel):
    id: str
    t: int
    entity_id: str
    metric: str
    score: float
    confidence: float
    severity: str
    explanation: str
    persistence: int = 0


class Hypothesis(BaseModel):
    id: str
    title: str
    score: float
    supporting: int
    contradicting: int
    explanation: str


class RiskAssessment(BaseModel):
    t: int
    entity_id: str
    score: float
    components: Dict[str, float]
    explanation: str


class Action(BaseModel):
    id: str
    t: int
    kind: str
    target: str
    expected_effect: str
    autonomy: str
    status: str
    simulator_only: bool = True


class VerificationResult(BaseModel):
    id: str
    action_id: str
    t: int
    success: bool
    observed_before: Dict[str, Any]
    observed_after: Dict[str, Any]
    proof: str


class EvidenceBundle(BaseModel):
    id: str
    t: int
    items: Dict[str, Any]
    sha256: str


class AuditEvent(BaseModel):
    id: str
    t: int
    actor: str
    event: str
    payload: Dict[str, Any]


# ------------------------------------------------------------
# SQLite persistence
# ------------------------------------------------------------

TABLES = [
    "observations",
    "entities",
    "states",
    "detections",
    "investigations",
    "hypotheses",
    "actions",
    "verifications",
    "evidence",
    "model_runs",
    "metrics",
    "audit_events",
]


@st.cache_resource
def get_db() -> sqlite3.Connection:
    db_path = os.getenv("MPM_DB", "mpm_demo.db")
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
    except Exception:
        conn = sqlite3.connect(":memory:", check_same_thread=False)

    cur = conn.cursor()
    for table in TABLES:
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, t INTEGER, payload TEXT)"
        )
    conn.commit()
    return conn


def clear_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    for table in TABLES:
        cur.execute(f"DELETE FROM {table}")
    conn.commit()


def insert_record(conn: sqlite3.Connection, table: str, rec_id: str, t: int, payload: Dict[str, Any]) -> None:
    conn.execute(
        f"INSERT OR REPLACE INTO {table} (id, t, payload) VALUES (?, ?, ?)",
        (rec_id, safe_int(t, 0), canonical_json(payload)),
    )


# ------------------------------------------------------------
# Synthetic world / scenario definitions
# ------------------------------------------------------------

ENTITIES: List[Dict[str, Any]] = [
    {"id": "proj-atlas", "type": "project", "criticality": 1.00},
    {"id": "agent-planner", "type": "agent", "criticality": 0.75},
    {"id": "agent-coder", "type": "agent", "criticality": 0.90},
    {"id": "agent-reviewer", "type": "agent", "criticality": 0.80},
    {"id": "agent-qa", "type": "agent", "criticality": 0.85},
    {"id": "agent-deployer", "type": "agent", "criticality": 0.70},
]

ENTITY_CRITICALITY = {entity["id"]: float(entity["criticality"]) for entity in ENTITIES}

METRICS: Dict[str, Dict[str, Any]] = {
    "throughput": {"baseline": 42.0, "amp": 3.0, "bad": "low", "min": 0.0},
    "cycle_time": {"baseline": 18.0, "amp": 1.2, "bad": "high", "min": 1.0},
    "defect_rate": {"baseline": 4.0, "amp": 0.4, "bad": "high", "min": 0.0},
    "test_pass_rate": {"baseline": 96.0, "amp": 0.6, "bad": "low", "min": 0.0, "max": 100.0},
    "budget_burn": {"baseline": 420.0, "amp": 18.0, "bad": "high", "min": 0.0},
    "incident_count": {"baseline": 0.25, "amp": 0.08, "bad": "high", "min": 0.0},
}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "dependency_deadlock": {
        "start": 20,
        "end": 110,
        "targets": ["agent-coder", "proj-atlas"],
        "effects": {"throughput": -16.0, "cycle_time": 8.0, "test_pass_rate": -4.0},
    },
    "flaky_tests": {
        "start": 24,
        "end": 108,
        "targets": ["agent-qa"],
        "effects": {"test_pass_rate": -9.0, "defect_rate": 5.0},
    },
    "budget_overrun": {
        "start": 30,
        "end": 105,
        "targets": ["proj-atlas"],
        "effects": {"budget_burn": 330.0},
    },
    "telemetry_loss": {
        "start": 34,
        "end": 46,
        "targets": ["agent-deployer"],
        "effects": {},
    },
    "contradictory_reporting": {
        "start": 38,
        "end": 58,
        "targets": ["agent-reviewer"],
        "effects": {"defect_rate": 2.0},
    },
    "security_incident": {
        "start": 50,
        "end": 100,
        "targets": ["agent-deployer"],
        "effects": {"incident_count": 3.2, "cycle_time": 2.0},
    },
}


class SyntheticWorld:
    """Deterministic seeded simulator. All autonomous actions remain inside this simulator."""

    def __init__(self, seed: int, config: Dict[str, Any]):
        self.seed = safe_int(seed, 42)
        self.config = config
        self.mitigations: Dict[str, float] = {}
        self.manual_until: Dict[str, int] = {}
        self.action_applied = False
        self.action_t: Optional[int] = None

    def _rng(self, *parts: Any) -> random.Random:
        key = "|".join([str(self.seed)] + [str(p) for p in parts])
        return random.Random(key)

    def inject(self, name: str, t: int, duration: int = 25) -> None:
        if name in SCENARIOS:
            self.manual_until[name] = max(
                safe_int(self.manual_until.get(name), -1),
                safe_int(t, 0) + safe_int(duration, 25),
            )

    def is_active(self, name: str, t: int) -> bool:
        scenario = SCENARIOS[name]
        normal = scenario["start"] <= t <= scenario["end"]
        manual = t <= safe_int(self.manual_until.get(name), -1)
        return normal or manual

    def effective_factor(self, name: str, t: int) -> float:
        if not self.is_active(name, t):
            return 0.0
        return max(0.0, 1.0 - safe_float(self.mitigations.get(name), 0.0))

    def active_scenario_names(self, t: int) -> List[str]:
        return [name for name in SCENARIOS if self.effective_factor(name, t) > 0.15]

    def truth_active(self, t: int) -> bool:
        return len(self.active_scenario_names(t)) > 0

    def contradiction_events(self, t: int) -> List[Dict[str, Any]]:
        if self.effective_factor("contradictory_reporting", t) > 0.15:
            return [
                {
                    "t": t,
                    "entity_id": "agent-reviewer",
                    "type": "contradictory_status",
                    "detail": "Reported review status is complete while defect-related signals are rising.",
                }
            ]
        return []

    def apply_action(self, action: Dict[str, Any], t: int) -> Dict[str, Any]:
        self.action_applied = True
        self.action_t = safe_int(t, 0)
        self.mitigations["dependency_deadlock"] = 0.85
        self.mitigations["budget_overrun"] = 0.75
        self.mitigations["flaky_tests"] = 0.35
        return {
            "status": "simulated",
            "bounded": True,
            "simulator_only": True,
            "action_kind": action.get("kind"),
        }

    def step(self, t: int) -> List[Observation]:
        observations: List[Observation] = []
        active_names = self.active_scenario_names(t)

        for entity in ENTITIES:
            entity_id = entity["id"]
            for metric, spec in METRICS.items():
                rng = self._rng("obs", entity_id, metric, t)
                phase = (str_hash(entity_id + ":" + metric) % 100) / 10.0
                base = spec["baseline"] + spec["amp"] * math.sin(t / 16.0 + phase)
                noise = rng.gauss(
                    0.0,
                    safe_float(self.config.get("noise_scale"), 0.05) * max(1.0, abs(spec["baseline"])),
                )
                value = base + noise

                for name, scenario in SCENARIOS.items():
                    factor = self.effective_factor(name, t)
                    if factor > 0.0 and entity_id in scenario["targets"] and metric in scenario["effects"]:
                        value += safe_float(scenario["effects"][metric], 0.0) * factor

                if self.action_applied and self.action_t is not None and t >= self.action_t:
                    decay = max(0.0, 1.0 - (t - self.action_t) / 20.0)
                    if metric == "throughput" and entity_id in ["proj-atlas", "agent-coder"]:
                        value += 9.0 * decay
                    if metric == "cycle_time" and entity_id in ["proj-atlas", "agent-coder"]:
                        value -= 4.5 * decay
                    if metric == "budget_burn" and entity_id == "proj-atlas":
                        value -= 170.0 * decay

                if "min" in spec:
                    value = max(safe_float(spec["min"], 0.0), value)
                if "max" in spec:
                    value = min(safe_float(spec["max"], 100.0), value)

                quality = 1.0
                if self.effective_factor("telemetry_loss", t) > 0.15 and entity_id in SCENARIOS["telemetry_loss"]["targets"]:
                    missing_rng = self._rng("missing", t, entity_id, metric)
                    if missing_rng.random() < 0.55:
                        quality = 0.05

                obs = Observation(
                    id=det_id("obs", t, entity_id, metric),
                    t=t,
                    entity_id=entity_id,
                    metric=metric,
                    value=float(value),
                    quality=float(quality),
                    source="synthetic_world",
                    meta={"missing": quality < 0.25, "active_scenarios": active_names},
                )
                observations.append(obs)

        return observations

    def read_state(self, t: int) -> Dict[str, Any]:
        return {
            "t": safe_int(t, 0),
            "active_scenarios": self.active_scenario_names(t),
            "mitigations": dict(self.mitigations),
            "action_applied": self.action_applied,
            "action_t": self.action_t,
        }


# ------------------------------------------------------------
# Tracking engine
# ------------------------------------------------------------

class TrackingEngine:
    def __init__(self) -> None:
        self.series: Dict[str, deque] = {}
        self.missing_counts: Dict[str, int] = {}
        self.contradiction_count = 0

    def update(
        self,
        observations: List[Observation],
        t: int,
        contradictions: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        states: Dict[str, Dict[str, Any]] = {}
        self.contradiction_count += len(contradictions)

        for obs in observations:
            key = f"{obs.entity_id}:{obs.metric}"
            if key not in self.series:
                self.series[key] = deque(maxlen=90)
                self.missing_counts[key] = 0

            if obs.quality < 0.25:
                self.missing_counts[key] += 1
                value = float(self.series[key][-1]) if self.series[key] else float(obs.value)
                quality = 0.10
            else:
                self.series[key].append(float(obs.value))
                value = float(obs.value)
                quality = float(obs.quality)

            states[key] = {
                "entity_id": obs.entity_id,
                "metric": obs.metric,
                "value": value,
                "quality": quality,
                "n": len(self.series[key]),
                "missing": self.missing_counts[key],
                "t": safe_int(t, 0),
            }

        return states

    def get_series(self, key: str) -> List[float]:
        return list(self.series.get(key, []))


# ------------------------------------------------------------
# AI engine: deterministic baseline + calibrated confidence
# ------------------------------------------------------------

class AIEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.active_detections: Dict[str, Dict[str, Any]] = {}
        self.last_scores: Dict[str, float] = {}

    def _stats(self, key: str) -> tuple[Optional[float], Optional[float]]:
        baseline = self.baselines.get(key)
        if not baseline or baseline["n"] < 2:
            return None, None
        variance = max(baseline["m2"] / max(1, baseline["n"] - 1), 1e-6)
        return float(baseline["mean"]), float(math.sqrt(variance))

    def _update_baseline(self, key: str, value: float, score: float) -> None:
        if key not in self.baselines:
            self.baselines[key] = {"mean": float(value), "m2": 0.0, "n": 0}

        baseline = self.baselines[key]
        threshold = safe_float(self.config.get("detection_threshold"), 0.58)

        # Robust-ish baseline update: update freely during cold start, otherwise only when not strongly anomalous.
        if baseline["n"] < 20 or score < threshold * 0.8:
            baseline["n"] += 1
            delta = float(value) - baseline["mean"]
            baseline["mean"] += delta / baseline["n"]
            baseline["m2"] += delta * (float(value) - baseline["mean"])

    def detect(
        self,
        states: Dict[str, Dict[str, Any]],
        t: int,
        contradictions: List[Dict[str, Any]],
    ) -> List[Detection]:
        detections: List[Detection] = []
        threshold = safe_float(self.config.get("detection_threshold"), 0.58)

        for key, state in states.items():
            metric = state["metric"]
            if metric not in METRICS:
                continue

            spec = METRICS[metric]
            value = safe_float(state["value"], 0.0)
            mean, std = self._stats(key)

            if mean is None:
                self._update_baseline(key, value, 0.0)
                continue

            std_floor = 0.15 if metric == "incident_count" else 0.75
            z = (value - mean) / max(std, std_floor)
            bad_z = z if spec["bad"] == "high" else -z

            score = clamp(bad_z / 4.0) if bad_z > 0 else 0.0
            if safe_float(state["quality"], 1.0) < 0.25:
                score = max(0.15, score * 0.4)

            prev_score = safe_float(self.last_scores.get(key), 0.0)
            score = clamp(0.65 * score + 0.35 * prev_score)
            self.last_scores[key] = score
            self._update_baseline(key, value, score)

            confidence_raw = min(1.0, safe_float(state["n"], 0.0) / 25.0) * (
                0.55 + 0.45 * safe_float(state["quality"], 0.5)
            )
            confidence = sigmoid(8.0 * (confidence_raw - 0.5))

            active = self.active_detections.get(key, {})
            persistence = safe_int(active.get("persistence"), 0)

            if score >= threshold:
                persistence += 1
            else:
                persistence = max(0, persistence - 1)

            if persistence >= 2 and score >= threshold:
                explanation = (
                    f"{metric} is {value:.2f} vs baseline {mean:.2f} ± {std:.2f}; "
                    f"z={z:+.2f}, bad_z={bad_z:+.2f}, persistence={persistence}, quality={state['quality']:.2f}."
                )
                detections.append(
                    Detection(
                        id=det_id("det", t, key, persistence),
                        t=t,
                        entity_id=state["entity_id"],
                        metric=metric,
                        score=round(float(score), 4),
                        confidence=round(float(confidence), 4),
                        severity="high" if score > 0.80 else "medium" if score > 0.65 else "low",
                        explanation=explanation,
                        persistence=persistence,
                    )
                )
                active["persistence"] = persistence
                active.setdefault("since", t)
                self.active_detections[key] = active
            else:
                if persistence == 0:
                    self.active_detections.pop(key, None)
                else:
                    active["persistence"] = persistence
                    self.active_detections[key] = active

        missing_total = sum(safe_int(state.get("missing"), 0) for state in states.values())
        if missing_total >= 10 and states:
            worst_key, worst_state = max(states.items(), key=lambda item: safe_int(item[1].get("missing"), 0))
            detections.append(
                Detection(
                    id=det_id("det_missing", t, worst_key),
                    t=t,
                    entity_id=worst_state["entity_id"],
                    metric="telemetry_quality",
                    score=round(clamp(0.50 + missing_total / 80.0, 0.55, 0.85), 4),
                    confidence=0.72,
                    severity="medium",
                    explanation=f"Missing or low-quality telemetry count is {missing_total}; recovery may be delayed.",
                    persistence=3,
                )
            )

        for idx, contradiction in enumerate(contradictions):
            detections.append(
                Detection(
                    id=det_id("det_contra", t, contradiction.get("entity_id", "unknown"), idx),
                    t=t,
                    entity_id=contradiction.get("entity_id", "unknown"),
                    metric="status_consistency",
                    score=0.72,
                    confidence=0.82,
                    severity="medium",
                    explanation=contradiction.get("detail", "Contradictory observations detected."),
                    persistence=2,
                )
            )

        return detections


# ------------------------------------------------------------
# Correlation / investigation engine
# ------------------------------------------------------------

class CorrelationEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def investigate(
        self,
        t: int,
        detections: List[Detection],
        states: Dict[str, Dict[str, Any]],
        world: SyntheticWorld,
    ) -> Dict[str, Any]:
        dmap = {f"{d.entity_id}:{d.metric}": d for d in detections}
        hypotheses: List[Hypothesis] = []

        def add_hypothesis(title: str, score: float, supporting: List[str], contradicting: List[str]) -> None:
            if score <= 0.20:
                return
            explanation = "Support: " + "; ".join(supporting) if supporting else "Support: weak signal."
            if contradicting:
                explanation += " | Contradicting: " + "; ".join(contradicting)
            hypotheses.append(
                Hypothesis(
                    id=det_id("hyp", t, title),
                    title=title,
                    score=round(clamp(score), 4),
                    supporting=len(supporting),
                    contradicting=len(contradicting),
                    explanation=explanation,
                )
            )

        # Dependency deadlock / integration bottleneck
        score = 0.0
        support: List[str] = []
        contra: List[str] = []
        if "agent-coder:throughput" in dmap or "proj-atlas:throughput" in dmap:
            score += 0.38
            support.append("Throughput dropped on delivery path")
        if "agent-coder:cycle_time" in dmap or "proj-atlas:cycle_time" in dmap:
            score += 0.32
            support.append("Cycle time increased")
        if "agent-qa:test_pass_rate" in dmap:
            score += 0.12
            support.append("Test pass rate pressure is correlated")
        if not support:
            contra.append("No strongly correlated delivery metrics detected")
        add_hypothesis("Dependency deadlock / integration bottleneck", score, support, contra)

        # Flaky tests
        score = 0.0
        support = []
        contra = []
        if "agent-qa:test_pass_rate" in dmap:
            score += 0.48
            support.append("QA test pass rate dropped")
        if "agent-qa:defect_rate" in dmap:
            score += 0.35
            support.append("Defect rate rose in QA")
        if "agent-reviewer:defect_rate" in dmap:
            score += 0.12
            support.append("Review-side defect signal is correlated")
        if not support:
            contra.append("No QA instability detected")
        add_hypothesis("Flaky tests / QA instability", score, support, contra)

        # Budget overrun
        score = 0.0
        support = []
        contra = []
        if "proj-atlas:budget_burn" in dmap:
            score += 0.82
            support.append("Budget burn is anomalously high")
        else:
            contra.append("Budget burn is not currently anomalous")
        add_hypothesis("Budget overrun / cost drift", score, support, contra)

        # Security incident
        score = 0.0
        support = []
        contra = []
        if "agent-deployer:incident_count" in dmap:
            score += 0.78
            support.append("Incident count spiked on deployer")
        if "agent-deployer:cycle_time" in dmap:
            score += 0.12
            support.append("Deployment cycle time is correlated")
        if not support:
            contra.append("No deployment/security anomaly detected")
        add_hypothesis("Security incident / deployment instability", score, support, contra)

        # Telemetry loss
        missing_total = sum(safe_int(state.get("missing"), 0) for state in states.values())
        support = []
        contra = []
        if missing_total > 8:
            support.append(f"Missing telemetry count is {missing_total}")
            score = clamp(0.35 + missing_total / 50.0, 0.35, 0.90)
        else:
            score = 0.0
            contra.append("Missing telemetry is within normal bounds")
        add_hypothesis("Telemetry loss / observability gap", score, support, contra)

        # Contradictory reporting
        score = 0.0
        support = []
        contra = []
        if any(d.metric == "status_consistency" for d in detections):
            score = 0.76
            support.append("Reported status contradicts measured quality signals")
        else:
            contra.append("No contradictory reporting detected")
        add_hypothesis("Contradictory reporting / status mismatch", score, support, contra)

        hypotheses.sort(key=lambda h: h.score, reverse=True)

        timeline: List[Dict[str, Any]] = []
        for detection in detections[:15]:
            timeline.append(
                {
                    "t": detection.t,
                    "type": "detection",
                    "summary": f"{detection.entity_id}:{detection.metric} score={detection.score:.2f}",
                }
            )
        for hypothesis in hypotheses[:6]:
            timeline.append(
                {
                    "t": t,
                    "type": "hypothesis",
                    "summary": f"{hypothesis.title} score={hypothesis.score:.2f}",
                }
            )

        supporting_total = sum(h.supporting for h in hypotheses)
        contradicting_total = sum(h.contradicting for h in hypotheses)
        evidence_quality = clamp(
            supporting_total / max(1.0, supporting_total + contradicting_total + 0.05 * missing_total + 1.0)
        )

        return {
            "t": safe_int(t, 0),
            "hypotheses": [model_to_dict(h) for h in hypotheses],
            "timeline": timeline,
            "evidence_quality": round(float(evidence_quality), 4),
            "missing_total": missing_total,
            "world_state": world.read_state(t),
        }


# ------------------------------------------------------------
# Risk engine
# ------------------------------------------------------------

class RiskEngine:
    def score(
        self,
        t: int,
        investigation: Dict[str, Any],
        detections: List[Detection],
        states: Dict[str, Dict[str, Any]],
        world: SyntheticWorld,
    ) -> RiskAssessment:
        if detections:
            anomaly = float(max(d.score for d in detections))
            confidence = float(max(d.confidence for d in detections))
            impact = float(max(ENTITY_CRITICALITY.get(d.entity_id, 0.5) for d in detections))
            persistence = float(max(safe_int(getattr(d, "persistence", 0), 0) for d in detections)) / 12.0
            persistence = clamp(persistence)
        else:
            anomaly = 0.05
            confidence = 0.20
            impact = 0.20
            persistence = 0.0

        evidence_quality = safe_float(investigation.get("evidence_quality"), 0.35)
        uncertainty = clamp(1.0 - 0.60 * confidence - 0.40 * evidence_quality)
        certainty = 1.0 - uncertainty

        score = 100.0 * (
            0.28 * anomaly
            + 0.20 * impact
            + 0.15 * persistence
            + 0.15 * confidence
            + 0.12 * evidence_quality
            + 0.10 * certainty
        )
        score = round(clamp(score, 0.0, 100.0), 2)

        components = {
            "anomaly_strength": round(anomaly, 4),
            "impact": round(impact, 4),
            "persistence": round(persistence, 4),
            "confidence": round(confidence, 4),
            "evidence_quality": round(evidence_quality, 4),
            "uncertainty": round(uncertainty, 4),
            "certainty": round(certainty, 4),
        }

        top_entity = detections[0].entity_id if detections else "proj-atlas"
        explanation = (
            f"Risk {score:.1f}/100 from anomaly={anomaly:.2f}, impact={impact:.2f}, persistence={persistence:.2f}, "
            f"confidence={confidence:.2f}, evidence_quality={evidence_quality:.2f}, uncertainty={uncertainty:.2f}."
        )

        return RiskAssessment(
            t=safe_int(t, 0),
            entity_id=top_entity,
            score=score,
            components=components,
            explanation=explanation,
        )


# ------------------------------------------------------------
# Policy gate
# ------------------------------------------------------------

class PolicyEngine:
    BLOCKED_ACTIONS = {
        "delete_repository",
        "force_merge",
        "production_deploy",
        "disable_monitoring",
    }

    def classify(self, action: Dict[str, Any], risk_score: float, config: Dict[str, Any]) -> str:
        if action.get("kind") in self.BLOCKED_ACTIONS:
            return "block"

        autonomy_mode = config.get("autonomy_mode", "safe_simulation")
        if autonomy_mode == "observe":
            return "observe"
        if autonomy_mode == "recommend_only":
            return "recommend"
        if autonomy_mode == "manual":
            return "require_approval"

        risk_score = safe_float(risk_score, 0.0)
        if risk_score < 30:
            return "observe"
        if risk_score < 50:
            return "recommend"
        if action.get("reversible") and action.get("bounded") and risk_score < 95:
            return "simulate"
        return "require_approval"


# ------------------------------------------------------------
# Independent verification engine
# ------------------------------------------------------------

class VerificationEngine:
    def verify(
        self,
        action_payload: Dict[str, Any],
        before: Dict[str, Any],
        world: SyntheticWorld,
        tracker: TrackingEngine,
        t: int,
    ) -> VerificationResult:
        after_state = world.read_state(t)

        def latest(key: str) -> Optional[float]:
            series = tracker.get_series(key)
            return float(series[-1]) if series else None

        after = {
            "t": safe_int(t, 0),
            "active_scenarios": after_state["active_scenarios"],
            "throughput": latest("proj-atlas:throughput"),
            "cycle_time": latest("proj-atlas:cycle_time"),
            "mitigations": after_state["mitigations"],
        }

        proofs: List[str] = []
        success = False

        throughput_before = safe_float(before.get("throughput"), 0.0)
        throughput_after = safe_float(after.get("throughput"), 0.0)
        if throughput_before and throughput_after and throughput_after > throughput_before * 1.015:
            success = True
            proofs.append(f"throughput increased from {throughput_before:.1f} to {throughput_after:.1f}")

        cycle_before = safe_float(before.get("cycle_time"), 0.0)
        cycle_after = safe_float(after.get("cycle_time"), 0.0)
        if cycle_before and cycle_after and cycle_after < cycle_before * 0.985:
            success = True
            proofs.append(f"cycle_time decreased from {cycle_before:.1f} to {cycle_after:.1f}")

        if len(after["active_scenarios"]) < len(before.get("active_scenarios", [])):
            success = True
            proofs.append("active anomaly scenarios reduced")

        proof = "; ".join(proofs) if proofs else "No independent improvement observed yet."

        return VerificationResult(
            id=det_id("verify", action_payload.get("id", "action"), t),
            action_id=str(action_payload.get("id", "unknown")),
            t=safe_int(t, 0),
            success=success,
            observed_before=before,
            observed_after=after,
            proof=proof,
        )


# ------------------------------------------------------------
# Evidence engine
# ------------------------------------------------------------

class EvidenceEngine:
    def record(
        self,
        t: int,
        action_payload: Dict[str, Any],
        verification: VerificationResult,
        investigation: Optional[Dict[str, Any]],
        risk: Optional[RiskAssessment],
        world: SyntheticWorld,
        tracker: TrackingEngine,
    ) -> EvidenceBundle:
        tracker_sample = {
            key: list(series)[-6:]
            for key, series in list(tracker.series.items())[:10]
        }

        items = {
            "t": safe_int(t, 0),
            "action": action_payload,
            "verification": model_to_dict(verification),
            "top_hypotheses": (investigation or {}).get("hypotheses", [])[:3],
            "risk": model_to_dict(risk) if risk else {},
            "world_state": world.read_state(t),
            "tracker_sample": tracker_sample,
            "model_version": MODEL_VERSION,
            "config_version": CONFIG_VERSION,
        }
        digest = sha256_json(items)
        return EvidenceBundle(
            id=det_id("evidence", t, digest[:12]),
            t=safe_int(t, 0),
            items=items,
            sha256=digest,
        )


# ------------------------------------------------------------
# KPI summary
# ------------------------------------------------------------

def summarize_kpi(kpi: Dict[str, Any], inference_latency: List[float]) -> Dict[str, Any]:
    total = kpi["tp"] + kpi["fp"] + kpi["fn"] + kpi["tn"]
    accuracy = (kpi["tp"] + kpi["tn"]) / max(1, total)
    latency = mean_or_zero(kpi["latencies"])
    false_alert_denominator = max(1, kpi["normal_ticks"] + kpi["false_alert_ticks"])
    false_alert_rate = kpi["false_alert_ticks"] / false_alert_denominator
    continuity = mean_or_zero(kpi["continuity"])
    avg_inference_ms = mean_or_zero(inference_latency[-200:])
    verification_attempts = max(1, kpi["verification_attempts"])
    verification_success_rate = kpi["verification_success"] / verification_attempts
    avg_risk = kpi["risk_sum"] / max(1, kpi["risk_ticks"])

    return {
        "total_ticks": total,
        "accuracy": round(accuracy, 4),
        "tp": kpi["tp"],
        "fp": kpi["fp"],
        "fn": kpi["fn"],
        "tn": kpi["tn"],
        "detection_latency_ticks": round(latency, 2),
        "false_alert_rate": round(false_alert_rate, 4),
        "continuity": round(continuity, 4),
        "avg_inference_ms": round(avg_inference_ms, 3),
        "verification_success_rate": round(verification_success_rate, 4),
        "avg_risk": round(avg_risk, 2),
    }


# ------------------------------------------------------------
# Multi-agent PM system / closed loop
# ------------------------------------------------------------

class MultiAgentPMSystem:
    def __init__(self, config: Dict[str, Any], conn: sqlite3.Connection):
        self.config = dict(config)
        self.conn = conn
        self.t = 0
        self.seed = safe_int(config.get("seed"), 42)

        self.world = SyntheticWorld(self.seed, self.config)
        self.tracker = TrackingEngine()
        self.ai = AIEngine(self.config)
        self.correlator = CorrelationEngine(self.config)
        self.risk_engine = RiskEngine()
        self.policy = PolicyEngine()
        self.verifier = VerificationEngine()
        self.evidence_engine = EvidenceEngine()

        self.frames: List[Dict[str, Any]] = []
        self.last_detections: List[Detection] = []
        self.last_investigation: Optional[Dict[str, Any]] = None
        self.last_risk: Optional[RiskAssessment] = None
        self.last_states: Dict[str, Dict[str, Any]] = {}
        self.last_action: Optional[Action] = None
        self.last_verification: Optional[VerificationResult] = None

        self.pending_verification: Optional[Dict[str, Any]] = None
        self.pending_approval: Optional[Dict[str, Any]] = None
        self.action_done = False
        self.audit_seq = 0

        self.kpi: Dict[str, Any] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "latencies": [],
            "continuity": [],
            "false_alert_ticks": 0,
            "normal_ticks": 0,
            "risk_sum": 0.0,
            "risk_ticks": 0,
            "verification_attempts": 0,
            "verification_success": 0,
            "anomaly_start": None,
            "episode_detected": False,
            "last_truth_t": -10,
        }
        self.inference_latency: List[float] = []

        self.record_model_run()
        self.persist_entities()
        self.audit("system", "initialized", {"seed": self.seed, "model_version": MODEL_VERSION})

    def record_model_run(self) -> None:
        insert_record(
            self.conn,
            "model_runs",
            det_id("model_run", self.seed, MODEL_VERSION),
            0,
            {
                "model_version": MODEL_VERSION,
                "config_version": CONFIG_VERSION,
                "seed": self.seed,
                "method": "EWMA/z-score baseline + deterministic correlation + explainable weighted risk",
                "calibration": "sigmoid confidence calibration and persistence smoothing",
                "cold_start": "first 20 observations establish baseline",
                "created_at": time.time(),
            },
        )

    def persist_entities(self) -> None:
        for entity in ENTITIES:
            insert_record(self.conn, "entities", det_id("entity", entity["id"]), 0, entity)

    def audit(self, actor: str, event: str, payload: Dict[str, Any]) -> None:
        self.audit_seq += 1
        record = AuditEvent(
            id=det_id("audit", self.t, event, self.audit_seq),
            t=self.t,
            actor=actor,
            event=event,
            payload=payload,
        )
        insert_record(self.conn, "audit_events", record.id, self.t, model_to_dict(record))

    def inject_scenario(self, name: str, duration: int = 25) -> None:
        self.world.inject(name, self.t, duration)
        self.audit("simulator", "manual_scenario_injected", {"scenario": name, "duration": duration})

    def capture_before(self, t: int, risk: Optional[RiskAssessment]) -> Dict[str, Any]:
        def latest(key: str) -> Optional[float]:
            series = self.tracker.get_series(key)
            return float(series[-1]) if series else None

        return {
            "t": safe_int(t, 0),
            "active_scenarios": self.world.active_scenario_names(t),
            "throughput": latest("proj-atlas:throughput"),
            "cycle_time": latest("proj-atlas:cycle_time"),
            "risk_score": float(risk.score if risk else 0.0),
        }

    def should_act(self, t: int, risk: RiskAssessment) -> bool:
        if self.action_done:
            return False
        if self.config.get("demo_mode") and t == 105:
            return True
        return (
            90 <= t <= 115
            and risk.score >= safe_float(self.config.get("risk_action_threshold"), 45.0)
        )

    def build_action(self, t: int, risk: RiskAssessment) -> Dict[str, Any]:
        return {
            "kind": "simulate_throttle_noncritical_tasks",
            "target": "proj-atlas",
            "expected_effect": "Reduce dependency contention and budget burn while protecting the critical path.",
            "reversible": True,
            "bounded": True,
            "simulator_only": True,
            "risk_at_proposal": risk.score,
            "t": safe_int(t, 0),
        }

    def step(self) -> Dict[str, Any]:
        start = time.perf_counter()
        t = self.t

        contradictions = self.world.contradiction_events(t)
        observations = self.world.step(t)

        persist_verbose = t < safe_int(self.config.get("persist_max_ticks"), 1200)
        if persist_verbose:
            for obs in observations:
                insert_record(self.conn, "observations", obs.id, t, model_to_dict(obs))

        states = self.tracker.update(observations, t, contradictions)
        self.last_states = states

        if persist_verbose:
            insert_record(
                self.conn,
                "states",
                det_id("state", t),
                t,
                {"states": states, "contradiction_count": self.tracker.contradiction_count},
            )

        detections = self.ai.detect(states, t, contradictions)
        self.last_detections = detections

        for detection in detections:
            insert_record(self.conn, "detections", detection.id, t, model_to_dict(detection))

        if detections or contradictions:
            investigation = self.correlator.investigate(t, detections, states, self.world)
            insert_record(self.conn, "investigations", det_id("investigation", t), t, investigation)
            for hypothesis in investigation.get("hypotheses", []):
                insert_record(
                    self.conn,
                    "hypotheses",
                    hypothesis.get("id", det_id("hyp", t, hypothesis.get("title"))),
                    t,
                    hypothesis,
                )
        else:
            investigation = {
                "t": t,
                "hypotheses": [],
                "timeline": [],
                "evidence_quality": 0.35,
                "missing_total": 0,
                "world_state": self.world.read_state(t),
            }

        self.last_investigation = investigation
        risk = self.risk_engine.score(t, investigation, detections, states, self.world)
        self.last_risk = risk

        action_record: Optional[Action] = None
        if self.should_act(t, risk):
            action_dict = self.build_action(t, risk)
            policy_decision = self.policy.classify(action_dict, risk.score, self.config)

            action_record = Action(
                id=det_id("act", t, action_dict["kind"]),
                t=t,
                kind=action_dict["kind"],
                target=action_dict["target"],
                expected_effect=action_dict["expected_effect"],
                autonomy=policy_decision,
                status="proposed",
                simulator_only=True,
            )
            self.last_action = action_record
            insert_record(self.conn, "actions", action_record.id, t, model_to_dict(action_record))
            self.audit("agent", "action_proposed", model_to_dict(action_record))

            if policy_decision == "simulate" and self.config.get("autonomy_mode") == "safe_simulation":
                before = self.capture_before(t, risk)
                self.world.apply_action(action_dict, t)
                self.action_done = True
                self.pending_verification = {
                    "due": t + 5,
                    "action": model_to_dict(action_record),
                    "before": before,
                }
                action_record.status = "simulated"
                insert_record(self.conn, "actions", action_record.id, t, model_to_dict(action_record))
                self.audit("policy_gate", "action_simulated", {"action_id": action_record.id, "risk": risk.score})

            elif policy_decision == "require_approval":
                before = self.capture_before(t, risk)
                action_record.status = "pending_approval"
                self.pending_approval = {
                    "action_dict": action_dict,
                    "action_record": model_to_dict(action_record),
                    "before": before,
                }
                insert_record(self.conn, "actions", action_record.id, t, model_to_dict(action_record))
                self.audit("policy_gate", "approval_required", {"action_id": action_record.id, "risk": risk.score})

            else:
                action_record.status = f"policy_{policy_decision}"
                insert_record(self.conn, "actions", action_record.id, t, model_to_dict(action_record))
                self.audit(
                    "policy_gate",
                    f"action_{policy_decision}",
                    {"action_id": action_record.id, "risk": risk.score},
                )

        verification: Optional[VerificationResult] = None
        if self.pending_verification and t >= safe_int(self.pending_verification.get("due"), 0):
            verification = self.verifier.verify(
                self.pending_verification["action"],
                self.pending_verification["before"],
                self.world,
                self.tracker,
                t,
            )
            self.last_verification = verification
            insert_record(self.conn, "verifications", verification.id, t, model_to_dict(verification))

            evidence = self.evidence_engine.record(
                t=t,
                action_payload=self.pending_verification["action"],
                verification=verification,
                investigation=self.last_investigation,
                risk=self.last_risk,
                world=self.world,
                tracker=self.tracker,
            )
            insert_record(self.conn, "evidence", evidence.id, t, model_to_dict(evidence))

            self.kpi["verification_attempts"] += 1
            if verification.success:
                self.kpi["verification_success"] += 1

            self.audit(
                "verifier",
                "verification_completed",
                {"success": verification.success, "proof": verification.proof},
            )
            self.audit("evidence", "evidence_hashed", {"sha256": evidence.sha256})
            self.pending_verification = None

        self.update_kpi(t, detections, risk, states, start)
        self.frames.append(self.render_frame(t, states, detections, investigation, risk, action_record, verification))

        self.t += 1
        self.conn.commit()

        return {
            "t": t,
            "phase": phase_label(t),
            "risk": risk.score,
            "detections": len(detections),
        }

    def update_kpi(
        self,
        t: int,
        detections: List[Detection],
        risk: RiskAssessment,
        states: Dict[str, Dict[str, Any]],
        start: float,
    ) -> None:
        inference_ms = (time.perf_counter() - start) * 1000.0
        self.inference_latency.append(inference_ms)
        if len(self.inference_latency) > 1000:
            self.inference_latency = self.inference_latency[-1000:]

        truth = self.world.truth_active(t)
        detected = len(detections) > 0

        if truth:
            if self.kpi.get("anomaly_start") is None:
                self.kpi["anomaly_start"] = t
                self.kpi["episode_detected"] = False

            if detected and not self.kpi.get("episode_detected", False):
                anomaly_start = self.kpi.get("anomaly_start")
                if anomaly_start is None:
                    anomaly_start = t
                    self.kpi["anomaly_start"] = t

                self.kpi["latencies"].append(max(0, t - safe_int(anomaly_start, t)))
                self.kpi["episode_detected"] = True
        else:
            if detected:
                self.kpi["false_alert_ticks"] += 1
            else:
                self.kpi["normal_ticks"] += 1

            self.kpi["anomaly_start"] = None
            self.kpi["episode_detected"] = False

        if truth and detected:
            self.kpi["tp"] += 1
        elif truth and not detected:
            self.kpi["fn"] += 1
        elif not truth and detected:
            self.kpi["fp"] += 1
        else:
            self.kpi["tn"] += 1

        continuity = sum(1 for state in states.values() if safe_int(state.get("n"), 0) > 0) / max(1, len(states))
        self.kpi["continuity"].append(float(continuity))
        if len(self.kpi["continuity"]) > 1000:
            self.kpi["continuity"] = self.kpi["continuity"][-1000:]

        self.kpi["risk_sum"] += float(risk.score)
        self.kpi["risk_ticks"] += 1
        self.kpi["last_truth_t"] = t

        if t % 5 == 0:
            summary = summarize_kpi(self.kpi, self.inference_latency)
            insert_record(self.conn, "metrics", det_id("metric", t), t, summary)

    def render_frame(
        self,
        t: int,
        states: Dict[str, Dict[str, Any]],
        detections: List[Detection],
        investigation: Dict[str, Any],
        risk: RiskAssessment,
        action_record: Optional[Action],
        verification: Optional[VerificationResult],
    ) -> Dict[str, Any]:
        top_hypothesis = "None"
        if investigation and investigation.get("hypotheses"):
            top_hypothesis = investigation["hypotheses"][0].get("title", "None")

        action_status = "none"
        if action_record:
            action_status = action_record.status
        elif self.world.action_applied:
            action_status = "simulated"

        verification_status = ""
        if verification:
            verification_status = "success" if verification.success else "failed"

        log: List[str] = []
        for detection in detections[:3]:
            log.append(
                f"DETECT {detection.entity_id}:{detection.metric} score={detection.score:.2f} conf={detection.confidence:.2f}"
            )
        if top_hypothesis != "None":
            log.append(f"HYPOTHESIS {top_hypothesis}")
        if action_record:
            log.append(f"ACTION {action_record.kind} -> {action_record.autonomy}/{action_record.status}")
        if verification:
            log.append(f"VERIFY {'success' if verification.success else 'fail'}: {verification.proof[:90]}")
        if self.world.action_applied:
            log.append("Autonomy bounded to simulator-only; evidence hashed.")

        return {
            "t": safe_int(t, 0),
            "phase": phase_label(t),
            "risk_score": float(risk.score),
            "top_hypothesis": top_hypothesis,
            "active_scenarios": self.world.active_scenario_names(t),
            "detection_count": len(detections),
            "action_status": action_status,
            "verification_status": verification_status,
            "sparkline": self.tracker.get_series("proj-atlas:throughput")[-45:],
            "log": log,
        }

    def fast_forward(self, n: int) -> None:
        for _ in range(safe_int(n, 0)):
            self.step()


# ------------------------------------------------------------
# Video generation from simulation frames
# ------------------------------------------------------------

def render_pil_frame(frame: Dict[str, Any]) -> Image.Image:
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), "#0b1220")
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, VIDEO_WIDTH, 54], fill="#111827")
    d.text((16, 10), "Multi-Agent Project Manager - Agentic AI", fill="#e5e7eb")
    d.text((16, 30), f"t={frame.get('t', 0)}s | {frame.get('phase', '')}", fill="#93c5fd")

    risk = safe_float(frame.get("risk_score"), 0.0)
    d.rectangle([16, 70, 320, 170], outline="#334155")
    d.text((24, 78), "Explainable Risk", fill="#9ca3af")
    d.text((24, 96), f"{risk:.1f}/100", fill="#f8fafc")
    bar_color = "#22c55e" if risk < 40 else "#f59e0b" if risk < 70 else "#ef4444"
    d.rectangle([24, 130, 296, 150], outline="#475569")
    d.rectangle([26, 132, 26 + int(268 * clamp(risk / 100.0)), 148], fill=bar_color)

    d.rectangle([336, 70, VIDEO_WIDTH - 16, 170], outline="#334155")
    d.text((344, 78), "Top hypothesis", fill="#9ca3af")
    d.text((344, 98), str(frame.get("top_hypothesis", "None"))[:86], fill="#e5e7eb")
    d.text((344, 118), f"Scenarios: {', '.join(frame.get('active_scenarios', []))[:72]}", fill="#9ca3af")
    d.text(
        (344, 138),
        f"Detections: {frame.get('detection_count', 0)} | Action: {frame.get('action_status', 'none')}",
        fill="#9ca3af",
    )

    d.rectangle([16, 186, VIDEO_WIDTH - 16, 320], outline="#334155")
    d.text((24, 194), "proj-atlas throughput", fill="#9ca3af")
    values = frame.get("sparkline", [])
    if len(values) > 1:
        min_value = min(values)
        max_value = max(values)
        span = max(1e-6, max_value - min_value)
        points = []
        for i, value in enumerate(values):
            x = 24 + i * (900 / max(1, len(values) - 1))
            y = 310 - ((value - min_value) / span) * 100
            points.append((x, y))
        d.line(points, fill="#60a5fa", width=2)

    d.rectangle([16, 336, VIDEO_WIDTH - 16, VIDEO_HEIGHT - 16], outline="#334155")
    y = 346
    for line in frame.get("log", [])[:9]:
        d.text((24, y), str(line)[:120], fill="#cbd5e1")
        y += 20

    return img


def encode_mp4(arrays: List[np.ndarray], fps: int = VIDEO_FPS) -> bytes:
    import imageio

    temporary = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    path = temporary.name
    temporary.close()

    try:
        writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
            pixelformat="yuv420p",
        )
        for array in arrays:
            writer.append_data(array)
        writer.close()

        with open(path, "rb") as handle:
            return handle.read()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def encode_gif(arrays: List[np.ndarray], fps: int = VIDEO_FPS) -> bytes:
    images = [Image.fromarray(array) for array in arrays]
    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
        optimize=True,
    )
    return buffer.getvalue()


def ensure_video() -> None:
    if st.session_state.get("video_bytes") is not None or st.session_state.get("video_error") is not None:
        return

    system = st.session_state.get("system")
    if not system or len(system.frames) < 20:
        return

    with st.spinner("Rendering downloadable 60-second demo video from live simulation frames..."):
        demo_seconds = safe_int(st.session_state.config.get("demo_seconds"), 120)
        frames = system.frames[:demo_seconds]
        arrays = [np.array(render_pil_frame(frame), dtype=np.uint8) for frame in frames]

        try:
            video_bytes = encode_mp4(arrays, fps=VIDEO_FPS)
            st.session_state.video_bytes = video_bytes
            st.session_state.video_mime = "video/mp4"
            st.session_state.video_ext = "mp4"
            st.session_state.video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        except Exception as mp4_error:
            try:
                gif_bytes = encode_gif(arrays, fps=VIDEO_FPS)
                st.session_state.video_bytes = gif_bytes
                st.session_state.video_mime = "image/gif"
                st.session_state.video_ext = "gif"
                st.session_state.video_b64 = base64.b64encode(gif_bytes).decode("utf-8")
            except Exception as gif_error:
                st.session_state.video_error = f"MP4 error: {mp4_error}; GIF error: {gif_error}"


def render_video_panel() -> None:
    ensure_video()

    if st.session_state.get("video_bytes"):
        mime = st.session_state.get("video_mime", "application/octet-stream")
        ext = st.session_state.get("video_ext", "bin")
        seconds = safe_int(st.session_state.config.get("demo_seconds"), 120) // VIDEO_FPS

        st.download_button(
            label=f"Download {seconds}-second demo video ({ext.upper()})",
            data=st.session_state.video_bytes,
            file_name=f"mpm_demo.{ext}",
            mime=mime,
            key="download_demo_video",
        )

        if mime == "video/mp4" and st.session_state.get("video_b64"):
            components.html(
                f"<video controls autoplay muted loop width='100%' src='data:video/mp4;base64,{st.session_state.video_b64}'></video>",
                height=560,
            )
        elif mime == "image/gif":
            st.image(st.session_state.video_bytes, caption="Deterministic demo capture")
    elif st.session_state.get("video_error"):
        st.error(f"Video generation failed: {st.session_state.video_error}")
    else:
        st.info("Video is not ready yet.")


# ------------------------------------------------------------
# Session initialization
# ------------------------------------------------------------

def initialize_session() -> None:
    st.session_state.setdefault("config", DEFAULT_CONFIG.copy())

    demo_seconds = safe_int(st.session_state.config.get("demo_seconds"), 120)

    if st.session_state.get("reset_requested"):
        st.session_state.reset_requested = False
        clear_db(get_db())
        st.session_state.system = MultiAgentPMSystem(st.session_state.config, get_db())
        st.session_state.system.fast_forward(demo_seconds)
        st.session_state.video_bytes = None
        st.session_state.video_b64 = None
        st.session_state.video_error = None
        st.session_state.live = False
        st.session_state.live_ticks_remaining = 0
        st.session_state.replay_requested = False

    if "system" not in st.session_state:
        clear_db(get_db())
        st.session_state.system = MultiAgentPMSystem(st.session_state.config, get_db())
        st.session_state.system.fast_forward(demo_seconds)
        st.session_state.video_bytes = None
        st.session_state.video_b64 = None
        st.session_state.video_error = None
        st.session_state.live = False
        st.session_state.live_ticks_remaining = 0

    if st.session_state.get("auto_started") is None:
        st.session_state.auto_started = True
        st.session_state.live = True
        st.session_state.live_ticks_remaining = safe_int(st.session_state.config.get("live_ticks"), 25)


# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------

def detections_dataframe(system: MultiAgentPMSystem) -> pd.DataFrame:
    records = [model_to_dict(d) for d in system.last_detections]
    return pd.DataFrame(records)


def hypotheses_dataframe(system: MultiAgentPMSystem) -> pd.DataFrame:
    investigation = system.last_investigation or {}
    return pd.DataFrame(investigation.get("hypotheses", []))


def timeline_dataframe(system: MultiAgentPMSystem) -> pd.DataFrame:
    investigation = system.last_investigation or {}
    return pd.DataFrame(investigation.get("timeline", []))


def risk_time_dataframe(system: MultiAgentPMSystem) -> pd.DataFrame:
    if not system.frames:
        return pd.DataFrame(columns=["t", "risk_score", "detection_count"])
    return pd.DataFrame(system.frames)[["t", "risk_score", "detection_count"]]


def throughput_figure(system: MultiAgentPMSystem) -> go.Figure:
    series = system.tracker.get_series("proj-atlas:throughput")
    fig = go.Figure()
    if series:
        fig.add_trace(go.Scatter(x=list(range(len(series))), y=series, mode="lines", name="throughput"))
    fig.update_layout(height=280, margin={"l": 10, "r": 10, "t": 25, "b": 10}, title="proj-atlas throughput")
    return fig


def risk_figure(system: MultiAgentPMSystem) -> go.Figure:
    df = risk_time_dataframe(system)
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df["t"], y=df["risk_score"], mode="lines", name="risk"))
    fig.update_layout(height=280, margin={"l": 10, "r": 10, "t": 25, "b": 10}, title="Risk over time")
    return fig


def query_table(table: str, limit: int = 30) -> pd.DataFrame:
    conn = get_db()
    try:
        df = pd.read_sql_query(f"SELECT t, payload FROM {table} ORDER BY t DESC LIMIT {safe_int(limit, 30)}", conn)
    except Exception:
        df = pd.DataFrame(columns=["t", "payload"])
    return df


def render_sidebar(system: MultiAgentPMSystem) -> None:
    with st.sidebar:
        st.subheader("Simulation controls")
        st.write(f"Tick: **{system.t}**")
        st.write(f"Phase: **{phase_label(system.t)}**")
        st.write(f"Live: **{bool(st.session_state.get('live', False))}**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start", key="btn_start"):
                st.session_state.live = True
                st.session_state.live_ticks_remaining = safe_int(st.session_state.config.get("live_ticks"), 90)
                st.rerun()
        with col2:
            if st.button("Pause", key="btn_pause"):
                st.session_state.live = False
                st.session_state.live_ticks_remaining = 0
                st.rerun()

        col3, col4 = st.columns(2)
        with col3:
            if st.button("Reset", key="btn_reset"):
                st.session_state.reset_requested = True
                st.session_state.live = False
                st.rerun()
        with col4:
            if st.button("Replay", key="btn_replay"):
                st.session_state.reset_requested = True
                st.session_state.live = False
                st.session_state.replay_requested = True
                st.rerun()

        if st.button("Demo mode", key="btn_demo_mode"):
            st.session_state.config["demo_mode"] = True
            st.session_state.reset_requested = True
            st.session_state.live = False
            st.rerun()

        st.markdown("---")
        st.caption(
            "Safe autonomy boundary: all proposed actions are simulator-only. "
            "Independent verification re-reads simulated state."
        )


# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------

initialize_session()
system: MultiAgentPMSystem = st.session_state.system
render_sidebar(system)

st.title("Multi-Agent Project Manager")
st.caption(
    "Agentic AI prototype: persistent state, contextual correlation, explainable risk, safe autonomy, "
    "independent verification, hashed evidence, and auditability. All data is synthetic and deterministic."
)

tab_overview, tab_live, tab_ai, tab_investigation, tab_action, tab_verification, tab_evidence, tab_analytics, tab_simulator, tab_config = st.tabs(
    [
        "Overview",
        "Simulation / Live",
        "AI Results",
        "Investigation",
        "Action",
        "Verification",
        "Evidence & Audit",
        "Analytics / KPIs",
        "Simulator",
        "Configuration",
    ]
)

with tab_overview:
    left, right = st.columns([2, 3])

    with left:
        kpi = summarize_kpi(system.kpi, system.inference_latency)
        st.metric("Current tick", system.t)
        st.metric("Phase", phase_label(system.t))
        st.metric("Risk", f"{system.last_risk.score if system.last_risk else 0:.1f}")
        st.metric("Detections", len(system.last_detections))
        st.metric("Accuracy", f"{kpi['accuracy'] * 100:.1f}%")
        st.metric("Detection latency", f"{kpi['detection_latency_ticks']:.1f} ticks")
        st.metric("False alert rate", f"{kpi['false_alert_rate'] * 100:.1f}%")
        st.metric("Verification success", f"{kpi['verification_success_rate'] * 100:.0f}%")

        st.subheader("Live demo video")
        render_video_panel()

    with right:
        st.plotly_chart(
            risk_figure(system),
            use_container_width=True,
            key="overview_risk_chart",
        )
        st.plotly_chart(
            throughput_figure(system),
            use_container_width=True,
            key="overview_throughput_chart",
        )

        active_scenarios = system.world.active_scenario_names(system.t)
        st.info("Active synthetic scenarios: " + (", ".join(active_scenarios) if active_scenarios else "none"))

        if system.last_risk:
            st.success(system.last_risk.explanation)

with tab_live:
    st.subheader("Live telemetry")
    st.plotly_chart(
        throughput_figure(system),
        use_container_width=True,
        key="live_throughput_chart",
    )

    detection_df = detections_dataframe(system)
    if detection_df.empty:
        st.write("No active detections.")
    else:
        st.dataframe(detection_df, use_container_width=True, key="live_detections_table")

    st.subheader("Recent simulation frames")
    frame_rows = system.frames[-12:]
    if frame_rows:
        frame_df = pd.DataFrame(frame_rows)[
            ["t", "phase", "risk_score", "detection_count", "top_hypothesis", "action_status"]
        ]
        st.dataframe(frame_df, use_container_width=True, key="live_frames_table")
    else:
        st.write("No simulation frames yet.")

with tab_ai:
    st.subheader("Model metadata")
    st.json(
        {
            "model_version": MODEL_VERSION,
            "config_version": CONFIG_VERSION,
            "method": "Deterministic EWMA/z-score baseline with persistence smoothing",
            "calibration": "Sigmoid confidence calibration based on sample size and telemetry quality",
            "cold_start": "First 20 observations establish baseline",
            "safe_autonomy": "simulator-only",
        }
    )

    st.subheader("Detections")
    detection_df = detections_dataframe(system)
    if detection_df.empty:
        st.write("No detections in the latest tick.")
    else:
        st.dataframe(detection_df, use_container_width=True, key="ai_detections_table")

    st.subheader("Risk components")
    if system.last_risk:
        st.json(system.last_risk.components)
    else:
        st.write("No risk assessment yet.")

with tab_investigation:
    st.subheader("Ranked hypotheses")
    hypothesis_df = hypotheses_dataframe(system)
    if hypothesis_df.empty:
        st.write("No active investigation.")
    else:
        st.dataframe(hypothesis_df, use_container_width=True, key="investigation_hypotheses_table")

    st.subheader("Investigation timeline")
    timeline_df = timeline_dataframe(system)
    if timeline_df.empty:
        st.write("No timeline events.")
    else:
        st.dataframe(timeline_df, use_container_width=True, key="investigation_timeline_table")

    if system.last_investigation:
        st.write(f"Evidence quality: {safe_float(system.last_investigation.get('evidence_quality'), 0.0):.2f}")
        st.write(f"Missing telemetry total: {safe_int(system.last_investigation.get('missing_total'), 0)}")

with tab_action:
    st.subheader("Proposed action and policy gate")

    if system.last_action:
        st.json(model_to_dict(system.last_action))
    else:
        st.write("No action proposed yet.")

    if system.pending_approval:
        st.warning("A high-risk action requires approval. In this prototype, approval remains simulator-only.")
        if st.button("Approve simulated action", key="btn_approve_simulated_action"):
            action_dict = system.pending_approval["action_dict"]
            before = system.pending_approval["before"]
            action_record_payload = system.pending_approval["action_record"]

            system.world.apply_action(action_dict, system.t)
            system.action_done = True
            system.pending_verification = {
                "due": system.t + 5,
                "action": action_record_payload,
                "before": before,
            }
            system.pending_approval = None
            system.audit("human", "approved_simulated_action", action_record_payload)
            st.rerun()
    else:
        st.success("No pending approval. Default demo action is policy-gated as a bounded simulation.")

with tab_verification:
    st.subheader("Independent verification")
    verification_df = query_table("verifications", 25)
    if verification_df.empty:
        st.write("No verification results yet.")
    else:
        rows = []
        for _, row in verification_df.iterrows():
            payload = json.loads(row["payload"])
            rows.append(
                {
                    "t": row["t"],
                    "success": payload.get("success"),
                    "proof": payload.get("proof"),
                    "action_id": payload.get("action_id"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, key="verification_table")

    if system.last_verification:
        st.json(model_to_dict(system.last_verification))

with tab_evidence:
    st.subheader("Hashed evidence bundles")
    evidence_df = query_table("evidence", 20)
    if evidence_df.empty:
        st.write("No evidence bundles yet.")
    else:
        rows = []
        for _, row in evidence_df.iterrows():
            payload = json.loads(row["payload"])
            rows.append(
                {
                    "t": row["t"],
                    "sha256": payload.get("sha256"),
                    "action_kind": payload.get("items", {}).get("action", {}).get("kind"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, key="evidence_table")

    st.subheader("Audit trail")
    audit_df = query_table("audit_events", 50)
    if audit_df.empty:
        st.write("No audit events yet.")
    else:
        rows = []
        for _, row in audit_df.iterrows():
            payload = json.loads(row["payload"])
            rows.append(
                {
                    "t": row["t"],
                    "actor": payload.get("actor"),
                    "event": payload.get("event"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, key="audit_table")

with tab_analytics:
    st.subheader("KPI summary")
    kpi = summarize_kpi(system.kpi, system.inference_latency)
    st.json(kpi)

    df = risk_time_dataframe(system)
    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["t"], y=df["risk_score"], name="risk", mode="lines"))
        fig.add_trace(go.Scatter(x=df["t"], y=df["detection_count"], name="detections", yaxis="y2", mode="lines"))
        fig.update_layout(
            title="Risk and detections",
            yaxis=dict(title="Risk"),
            yaxis2=dict(title="Detections", overlaying="y", side="right"),
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True, key="analytics_risk_detections_chart")

    metrics_df = query_table("metrics", 100)
    if not metrics_df.empty:
        st.subheader("Persisted metric snapshots")
        rows = []
        for _, row in metrics_df.iterrows():
            payload = json.loads(row["payload"])
            payload["t"] = row["t"]
            rows.append(payload)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, key="analytics_metrics_table")

with tab_simulator:
    st.subheader("Manual anomaly injection")
    st.write("Injected scenarios remain inside the simulator and are subject to the same policy and verification loop.")

    cols = st.columns(3)
    for idx, scenario_name in enumerate(SCENARIOS.keys()):
        with cols[idx % 3]:
            if st.button(f"Inject {scenario_name}", key=f"inject_{scenario_name}"):
                system.inject_scenario(scenario_name, duration=25)
                st.rerun()

    st.subheader("World state")
    st.json(system.world.read_state(system.t))

with tab_config:
    st.subheader("Configuration")
    cfg = st.session_state.config

    seed = st.number_input("Seed", value=safe_int(cfg.get("seed"), 42), step=1, key="config_seed")
    threshold = st.slider(
        "Detection threshold",
        0.30,
        0.90,
        safe_float(cfg.get("detection_threshold"), 0.58),
        0.01,
        key="config_detection_threshold",
    )
    risk_threshold = st.slider(
        "Risk action threshold",
        10.0,
        90.0,
        safe_float(cfg.get("risk_action_threshold"), 45.0),
        1.0,
        key="config_risk_action_threshold",
    )

    autonomy_options = ["observe", "recommend_only", "safe_simulation", "manual"]
    current_autonomy = cfg.get("autonomy_mode", "safe_simulation")
    if current_autonomy not in autonomy_options:
        current_autonomy = "safe_simulation"

    autonomy_mode = st.selectbox(
        "Autonomy mode",
        autonomy_options,
        index=autonomy_options.index(current_autonomy),
        key="config_autonomy_mode",
    )

    if st.button("Apply configuration and reset", key="btn_apply_config_reset"):
        st.session_state.config.update(
            {
                "seed": safe_int(seed, 42),
                "detection_threshold": safe_float(threshold, 0.58),
                "risk_action_threshold": safe_float(risk_threshold, 45.0),
                "autonomy_mode": autonomy_mode,
            }
        )
        st.session_state.reset_requested = True
        st.rerun()

    st.warning("Changing configuration resets the deterministic simulation and rebuilds the demo video.")


# ------------------------------------------------------------
# Optional live continuation
# ------------------------------------------------------------

if st.session_state.get("live", False) and st.session_state.get("live_ticks_remaining", 0) > 0:
    time.sleep(safe_float(st.session_state.config.get("live_sleep"), 0.2))
    st.session_state.system.step()
    st.session_state.live_ticks_remaining -= 1
    if st.session_state.live_ticks_remaining <= 0:
        st.session_state.live = False
    st.rerun()