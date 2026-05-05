"""
MedAI — Deduplication Engine (Phase 4)

Intelligent fact merging with five strategies:
    1. Exact duplicate detection          (merge + increment occurrences)
    2. Semantic duplicate detection       (alias / embedding similarity)
    3. Time-series detection              (same entity, different values, different dates)
    4. Conflict detection                 (EDGE CASE 1 — value/type/drug/temporal conflicts)
    5. Implausible-change detection       (medically impossible deltas)

All public methods accept and return plain ``dict`` "fact" objects rather than
``MKBRecord`` instances, so they can be called from extraction, enrichment,
ingestion, or tests without Pydantic friction. A canonical fact is:

    {
        "entity_type":  "diagnosis" | "medication" | "test_result" | ...,
        "entity_name":  str,
        "value":        float | str | None,
        "unit":         str | None,
        "date":         datetime.date | None,
        "subject":      "patient" | "father" | ...,
        "confidence":   float,
        "source":       str,
        ...
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional

from loguru import logger


# ── Terminology aliases (semantic equivalence) ───────────────────────────────
# Canonical term → set of equivalent surface forms (lowercased).
# Kept inline so the engine works without a database connection; the SQLite
# terminology_aliases table, if present, is consulted first via
# ``set_alias_source``.

_DEFAULT_ALIASES: dict[str, set[str]] = {
    "hypertension":        {"hypertension", "htn", "high blood pressure", "elevated bp",
                            "elevated blood pressure", "raised blood pressure"},
    "diabetes mellitus":   {"diabetes", "dm", "diabetes mellitus", "t2dm", "t1dm",
                            "type 2 diabetes", "type 1 diabetes"},
    "myocardial infarction": {"myocardial infarction", "mi", "ami", "heart attack",
                              "acute myocardial infarction"},
    "cerebrovascular accident": {"cva", "stroke", "cerebrovascular accident"},
    "chronic kidney disease": {"ckd", "chronic kidney disease", "chronic renal failure"},
    "congestive heart failure": {"chf", "congestive heart failure", "heart failure"},
    "hba1c":               {"hba1c", "a1c", "glycated haemoglobin",
                            "glycated hemoglobin", "glycosylated haemoglobin",
                            "glycosylated hemoglobin"},
    "hemoglobin":          {"hemoglobin", "haemoglobin", "hgb", "hb"},
    "cholesterol":         {"cholesterol", "chol", "total cholesterol"},
    "ldl":                 {"ldl", "ldl cholesterol", "ldl-c", "low density lipoprotein"},
    "hdl":                 {"hdl", "hdl cholesterol", "hdl-c", "high density lipoprotein"},
    "blood pressure":      {"blood pressure", "bp"},
    "heart rate":          {"heart rate", "hr", "pulse"},
    "weight":              {"weight", "body weight", "bw"},
    "temperature":         {"temperature", "temp", "body temperature"},
}

# Known drug interaction pairs (normalized lowercase, order-independent).
_DRUG_INTERACTIONS: dict[frozenset[str], str] = {
    frozenset({"warfarin", "rivaroxaban"}):       "anticoagulant_duplication",
    frozenset({"warfarin", "apixaban"}):          "anticoagulant_duplication",
    frozenset({"warfarin", "dabigatran"}):        "anticoagulant_duplication",
    frozenset({"ssri", "mao inhibitor"}):         "serotonin_syndrome_risk",
    frozenset({"fluoxetine", "tranylcypromine"}): "serotonin_syndrome_risk",
    frozenset({"sertraline", "phenelzine"}):      "serotonin_syndrome_risk",
    frozenset({"clopidogrel", "omeprazole"}):     "reduced_antiplatelet_effect",
}

# Mutually-exclusive type pairs (same entity, incompatible variants).
_MUTUALLY_EXCLUSIVE_TYPES: list[tuple[set[str], set[str]]] = [
    ({"type 1 diabetes", "t1dm"},        {"type 2 diabetes", "t2dm"}),
    ({"left-sided", "left sided", "left"}, {"right-sided", "right sided", "right"}),
    ({"benign"},                         {"malignant"}),
    ({"acute"},                          {"chronic"}),
]


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class DedupResult:
    """Outcome of a deduplication lookup."""
    strategy:       str                       # exact|semantic|timeseries|conflict|implausible|none
    matched_fact:   Optional[dict[str, Any]] = None
    action:         str = "insert"            # insert|merge|append_timeline|quarantine
    reason:         str = ""
    details:        dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictRecord:
    """Conflict snapshot to quarantine."""
    conflict_type:  str     # value_mismatch|type_mismatch|drug_interaction|temporal|implausible_change
    severity:       str     # low|medium|high|critical
    fact1:          dict[str, Any]
    fact2:          dict[str, Any]
    reason:         str
    detected_at:    datetime = field(default_factory=datetime.utcnow)


# ── Main engine ──────────────────────────────────────────────────────────────

class DeduplicationEngine:
    """Entry point for merging incoming facts into the knowledge base."""

    # implausibility thresholds per (entity_name, days-window, delta)
    _WEIGHT_DELTA_7D  = 5.0   # kg
    _WEIGHT_DELTA_30D = 20.0  # kg
    _BP_DELTA_SAMEDAY = 30.0  # mmHg systolic
    _TEMP_DELTA_SAMEDAY = 2.0 # °C
    _HBA1C_DELTA_60D  = 2.0   # percentage points

    _EXACT_DATE_WINDOW_DAYS = 7

    def __init__(
        self,
        aliases: Optional[dict[str, set[str]]] = None,
        embedding_fn=None,
        embedding_threshold: float = 0.92,
    ):
        """
        :param aliases: optional override of terminology_aliases mapping.
        :param embedding_fn: callable(str) -> list[float] for semantic fallback.
        :param embedding_threshold: cosine similarity threshold for match.
        """
        self._aliases = aliases or _DEFAULT_ALIASES
        self._reverse_alias = self._build_reverse_alias(self._aliases)
        self._embed = embedding_fn
        self._embed_threshold = embedding_threshold

    # ── public orchestration ────────────────────────────────────────────────

    def deduplicate(
        self,
        new_fact: dict[str, Any],
        existing_facts: Iterable[dict[str, Any]],
    ) -> DedupResult:
        """Apply strategies in priority order and return the first non-trivial result."""
        existing = list(existing_facts)

        exact = self.find_exact_match(new_fact, existing)
        if exact is not None:
            return DedupResult(
                strategy="exact",
                matched_fact=exact,
                action="merge",
                reason="Exact duplicate — incremented occurrence count.",
            )

        conflict = self.detect_conflict(new_fact, existing)
        if conflict is not None:
            return DedupResult(
                strategy="conflict",
                matched_fact=conflict.fact2 if conflict.fact2 is not new_fact else conflict.fact1,
                action="quarantine",
                reason=conflict.reason,
                details={
                    "conflict_type": conflict.conflict_type,
                    "severity":      conflict.severity,
                },
            )

        ts = self.find_timeseries_match(new_fact, existing)
        if ts is not None:
            return DedupResult(
                strategy="timeseries",
                matched_fact=ts,
                action="append_timeline",
                reason="Same entity, different date — stored as time-series point.",
            )

        sem = self.find_semantic_match(new_fact, existing)
        if sem is not None:
            return DedupResult(
                strategy="semantic",
                matched_fact=sem,
                action="merge",
                reason="Semantically equivalent — linked via alias / embedding similarity.",
            )

        return DedupResult(strategy="none", action="insert", reason="No match found.")

    # ── Strategy 1: exact match ─────────────────────────────────────────────

    def find_exact_match(
        self,
        new_fact: dict[str, Any],
        existing_facts: Iterable[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Match on entity_type + entity_name + value with dates within 7 days."""
        new_type = (new_fact.get("entity_type") or "").lower()
        new_name = self._norm(new_fact.get("entity_name"))
        new_val  = self._norm_value(new_fact.get("value"))
        new_date = self._to_date(new_fact.get("date"))
        new_subject = (new_fact.get("subject") or "patient").lower()

        for f in existing_facts:
            if (f.get("entity_type") or "").lower() != new_type:
                continue
            if self._norm(f.get("entity_name")) != new_name:
                continue
            if self._norm_value(f.get("value")) != new_val:
                continue
            if (f.get("subject") or "patient").lower() != new_subject:
                continue
            if not self._dates_close(new_date, self._to_date(f.get("date")),
                                     self._EXACT_DATE_WINDOW_DAYS):
                continue
            return f
        return None

    # ── Strategy 2: semantic match ──────────────────────────────────────────

    def find_semantic_match(
        self,
        new_fact: dict[str, Any],
        existing_facts: Iterable[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Match aliased or embedding-similar entities of the same type."""
        new_type = (new_fact.get("entity_type") or "").lower()
        new_canon = self._canonical(new_fact.get("entity_name"))
        new_name = self._norm(new_fact.get("entity_name"))

        for f in existing_facts:
            if (f.get("entity_type") or "").lower() != new_type:
                continue

            if new_canon and new_canon == self._canonical(f.get("entity_name")):
                return f

            # Fuzzy string fallback for minor spelling differences (>= 0.9 ratio).
            ratio = SequenceMatcher(None, new_name, self._norm(f.get("entity_name"))).ratio()
            if ratio >= 0.92:
                return f

            # Embedding fallback if provided.
            if self._embed is not None:
                try:
                    v1 = self._embed(str(new_fact.get("entity_name") or ""))
                    v2 = self._embed(str(f.get("entity_name") or ""))
                    if self._cosine(v1, v2) >= self._embed_threshold:
                        return f
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning(f"Embedding comparison failed: {exc}")
        return None

    # ── Strategy 3: time-series ─────────────────────────────────────────────

    def find_timeseries_match(
        self,
        new_fact: dict[str, Any],
        existing_facts: Iterable[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Same canonical entity, different date -> value change over time."""
        new_type = (new_fact.get("entity_type") or "").lower()
        new_canon = self._canonical(new_fact.get("entity_name")) \
            or self._norm(new_fact.get("entity_name"))
        new_date = self._to_date(new_fact.get("date"))

        if new_date is None:
            return None

        best = None
        best_gap = -1
        for f in existing_facts:
            if (f.get("entity_type") or "").lower() != new_type:
                continue
            f_canon = self._canonical(f.get("entity_name")) or self._norm(f.get("entity_name"))
            if f_canon != new_canon:
                continue
            f_date = self._to_date(f.get("date"))
            if f_date is None:
                continue
            gap = abs((new_date - f_date).days)
            if gap <= self._EXACT_DATE_WINDOW_DAYS:
                # Falls into "exact" window; don't treat as time-series.
                continue
            if gap > best_gap:
                best = f
                best_gap = gap
        return best

    # ── Strategy 4: conflict detection ──────────────────────────────────────

    def detect_conflict(
        self,
        new_fact: dict[str, Any],
        existing_facts: Iterable[dict[str, Any]],
    ) -> Optional[ConflictRecord]:
        """Identify conflicts that require user review."""
        new_type = (new_fact.get("entity_type") or "").lower()
        new_canon = self._canonical(new_fact.get("entity_name")) \
            or self._norm(new_fact.get("entity_name"))
        new_name_norm = self._norm(new_fact.get("entity_name"))
        new_date = self._to_date(new_fact.get("date"))

        # --- Drug interaction scan (medications only) ---
        if new_type == "medication":
            for f in existing_facts:
                if (f.get("entity_type") or "").lower() != "medication":
                    continue
                pair = frozenset({new_name_norm, self._norm(f.get("entity_name"))})
                if pair in _DRUG_INTERACTIONS:
                    return ConflictRecord(
                        conflict_type="drug_interaction",
                        severity="critical",
                        fact1=f,
                        fact2=new_fact,
                        reason=f"Dangerous drug combination: {_DRUG_INTERACTIONS[pair]}",
                    )

        # Temporal impossibility scans across entity types (treatment vs diagnosis).
        for f in existing_facts:
            temporal = self._temporal_conflict(new_fact, f)
            if temporal is not None:
                return temporal

        for f in existing_facts:
            if (f.get("entity_type") or "").lower() != new_type:
                continue

            f_canon = self._canonical(f.get("entity_name")) \
                or self._norm(f.get("entity_name"))
            f_date = self._to_date(f.get("date"))

            # Type mismatch — mutually exclusive variants of the "same" concept.
            if self._is_mutually_exclusive(new_name_norm, self._norm(f.get("entity_name"))):
                return ConflictRecord(
                    conflict_type="type_mismatch",
                    severity="high",
                    fact1=f,
                    fact2=new_fact,
                    reason=(
                        f"Mutually exclusive variants: "
                        f"'{f.get('entity_name')}' vs '{new_fact.get('entity_name')}'"
                    ),
                )

            if f_canon != new_canon:
                continue

            # Value mismatch for overlapping dates.
            if (new_date is not None and f_date is not None
                    and abs((new_date - f_date).days) <= 1
                    and self._values_differ(new_fact.get("value"), f.get("value"))):
                return ConflictRecord(
                    conflict_type="value_mismatch",
                    severity=self._value_conflict_severity(new_type),
                    fact1=f,
                    fact2=new_fact,
                    reason=(
                        f"Same-day value mismatch for '{new_fact.get('entity_name')}': "
                        f"{f.get('value')} {f.get('unit') or ''} vs "
                        f"{new_fact.get('value')} {new_fact.get('unit') or ''}"
                    ),
                )

            # Implausible physiological change over time.
            if self.is_implausible_change(f, new_fact):
                return ConflictRecord(
                    conflict_type="implausible_change",
                    severity="high",
                    fact1=f,
                    fact2=new_fact,
                    reason=(
                        f"Medically implausible change for '{new_fact.get('entity_name')}': "
                        f"{f.get('value')}→{new_fact.get('value')} "
                        f"between {f_date} and {new_date}"
                    ),
                )

        return None

    # ── Strategy 5: implausibility ──────────────────────────────────────────

    def is_implausible_change(
        self,
        fact1: dict[str, Any],
        fact2: dict[str, Any],
    ) -> bool:
        """Return True if fact1→fact2 is medically implausible."""
        canon = self._canonical(fact1.get("entity_name")) or self._norm(fact1.get("entity_name"))
        v1 = self._as_number(fact1.get("value"))
        v2 = self._as_number(fact2.get("value"))
        if v1 is None or v2 is None:
            return False

        d1 = self._to_date(fact1.get("date"))
        d2 = self._to_date(fact2.get("date"))
        if d1 is None or d2 is None:
            return False
        gap = abs((d2 - d1).days)
        delta = abs(v2 - v1)

        if canon == "weight":
            if gap < 7 and delta > self._WEIGHT_DELTA_7D:
                return True
            if gap < 30 and delta > self._WEIGHT_DELTA_30D:
                return True
        elif canon == "blood pressure":
            if gap == 0 and delta > self._BP_DELTA_SAMEDAY:
                return True
        elif canon == "temperature":
            if gap == 0 and delta > self._TEMP_DELTA_SAMEDAY:
                return True
        elif canon == "hba1c":
            if gap < 60 and delta > self._HBA1C_DELTA_60D:
                return True
        return False

    # ── Internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_reverse_alias(aliases: dict[str, set[str]]) -> dict[str, str]:
        reverse: dict[str, str] = {}
        for canonical, variants in aliases.items():
            reverse[canonical.lower()] = canonical.lower()
            for v in variants:
                reverse[v.lower()] = canonical.lower()
        return reverse

    def _canonical(self, name: Any) -> str:
        if not name:
            return ""
        return self._reverse_alias.get(self._norm(name), "")

    @staticmethod
    def _norm(s: Any) -> str:
        if s is None:
            return ""
        return " ".join(str(s).lower().strip().split())

    @staticmethod
    def _norm_value(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return round(float(v), 4)
        return str(v).strip().lower()

    @staticmethod
    def _as_number(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_date(v: Any) -> Optional[date]:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        try:
            return datetime.fromisoformat(str(v)).date()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _dates_close(a: Optional[date], b: Optional[date], window: int) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return abs((a - b).days) <= window

    def _values_differ(self, a: Any, b: Any) -> bool:
        if a is None or b is None:
            return False
        na, nb = self._as_number(a), self._as_number(b)
        if na is not None and nb is not None:
            if max(abs(na), abs(nb)) == 0:
                return False
            return abs(na - nb) / max(abs(na), abs(nb), 1e-9) > 0.02  # 2% tolerance
        return self._norm_value(a) != self._norm_value(b)

    @staticmethod
    def _value_conflict_severity(entity_type: str) -> str:
        # Medication dose mismatches and lab values carry higher weight.
        if entity_type == "medication":
            return "critical"
        if entity_type == "test_result":
            return "high"
        return "medium"

    @staticmethod
    def _is_mutually_exclusive(a: str, b: str) -> bool:
        for left, right in _MUTUALLY_EXCLUSIVE_TYPES:
            if (any(tok in a for tok in left) and any(tok in b for tok in right)) \
               or (any(tok in b for tok in left) and any(tok in a for tok in right)):
                return True
        return False

    def _temporal_conflict(
        self,
        new_fact: dict[str, Any],
        existing: dict[str, Any],
    ) -> Optional[ConflictRecord]:
        """Detect events ordered impossibly (treatment before diagnosis, etc.)."""
        new_date = self._to_date(new_fact.get("date"))
        ex_date = self._to_date(existing.get("date"))
        if new_date is None or ex_date is None:
            return None

        new_type = (new_fact.get("entity_type") or "").lower()
        ex_type = (existing.get("entity_type") or "").lower()

        treatment_types = {"medication", "treatment", "procedure", "surgery"}
        diagnosis_types = {"diagnosis", "condition"}

        if new_type in treatment_types and ex_type in diagnosis_types and new_date < ex_date:
            return ConflictRecord(
                conflict_type="temporal",
                severity="medium",
                fact1=existing,
                fact2=new_fact,
                reason=(
                    f"Treatment '{new_fact.get('entity_name')}' dated {new_date} is "
                    f"before diagnosis '{existing.get('entity_name')}' dated {ex_date}."
                ),
            )
        if ex_type in treatment_types and new_type in diagnosis_types and ex_date < new_date:
            return ConflictRecord(
                conflict_type="temporal",
                severity="medium",
                fact1=existing,
                fact2=new_fact,
                reason=(
                    f"Treatment '{existing.get('entity_name')}' dated {ex_date} is "
                    f"before diagnosis '{new_fact.get('entity_name')}' dated {new_date}."
                ),
            )
        return None

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        num = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return num / (na * nb)
