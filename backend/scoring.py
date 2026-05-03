"""
scoring.py — Medical scoring algorithms
NEW feature: Shock Index, NEWS2, Pediatric vital adjustments
"""
from typing import Optional
from models import VitalSigns, MedicalScore

def compute_scores(vitals: Optional[VitalSigns], age: Optional[int] = None) -> MedicalScore:
    """Compute all medical scores from vital signs."""
    if not vitals:
        return MedicalScore()

    score = MedicalScore()
    score.critical_vitals = []
    score.pediatric_adjusted = (age is not None and age < 18)

    # ── Shock Index ───────────────────────────────────────────────────────────
    if vitals.hr and vitals.bp:
        systolic = _parse_systolic(vitals.bp)
        if systolic and systolic > 0:
            si = round(vitals.hr / systolic, 2)
            score.shock_index = si
            if si < 0.6:
                score.shock_class = "Normal"
            elif si < 1.0:
                score.shock_class = "Mild concern"
            elif si < 1.2:
                score.shock_class = "Moderate shock"
            else:
                score.shock_class = "Severe shock"
                score.critical_vitals.append(f"Shock Index {si} — SEVERE SHOCK")

    # ── Critical Vital Flags ──────────────────────────────────────────────────
    thresholds = _get_thresholds(age)

    if vitals.hr:
        if vitals.hr < thresholds["hr_low"]:
            score.critical_vitals.append(f"Bradycardia: HR {vitals.hr} bpm")
        elif vitals.hr > thresholds["hr_high"]:
            score.critical_vitals.append(f"Tachycardia: HR {vitals.hr} bpm")

    if vitals.spo2 and vitals.spo2 < 90:
        score.critical_vitals.append(f"Hypoxia: SpO2 {vitals.spo2}%")

    if vitals.rr:
        if vitals.rr < thresholds["rr_low"]:
            score.critical_vitals.append(f"Bradypnea: RR {vitals.rr}/min")
        elif vitals.rr > thresholds["rr_high"]:
            score.critical_vitals.append(f"Tachypnea: RR {vitals.rr}/min")

    if vitals.gcs and vitals.gcs <= 8:
        score.critical_vitals.append(f"Severe GCS: {vitals.gcs}/15 — intubation threshold")

    if vitals.bgl:
        if vitals.bgl < 60:
            score.critical_vitals.append(f"Hypoglycemia: BGL {vitals.bgl} mg/dL")
        elif vitals.bgl > 400:
            score.critical_vitals.append(f"Severe hyperglycemia: BGL {vitals.bgl} mg/dL")

    if vitals.bp:
        sys = _parse_systolic(vitals.bp)
        if sys:
            min_sbp = _pediatric_min_sbp(age) if age and age < 18 else 90
            if sys < min_sbp:
                score.critical_vitals.append(f"Hypotension: SBP {sys} mmHg")

    # ── NEWS2 Score ───────────────────────────────────────────────────────────
    news2 = _compute_news2(vitals)
    if news2 is not None:
        score.news2_score = news2
        if news2 <= 4:
            score.news2_risk = "Low"
        elif news2 <= 6:
            score.news2_risk = "Medium"
        else:
            score.news2_risk = "High — consider escalation"

    return score

def _parse_systolic(bp_str: str) -> Optional[int]:
    """Parse systolic from '120/80' string."""
    try:
        return int(bp_str.split("/")[0].strip())
    except (ValueError, AttributeError, IndexError):
        return None

def _pediatric_min_sbp(age: int) -> int:
    """Pediatric hypotension threshold: 70 + (2 × age)"""
    return 70 + (2 * age)

def _get_thresholds(age: Optional[int]) -> dict:
    """Age-adjusted vital sign thresholds."""
    if age and age < 1:
        return {"hr_low": 100, "hr_high": 160, "rr_low": 30, "rr_high": 60}
    elif age and age < 5:
        return {"hr_low": 80, "hr_high": 140, "rr_low": 20, "rr_high": 40}
    elif age and age < 12:
        return {"hr_low": 70, "hr_high": 120, "rr_low": 15, "rr_high": 30}
    else:
        return {"hr_low": 40, "hr_high": 180, "rr_low": 8, "rr_high": 36}

def _compute_news2(v: VitalSigns) -> Optional[int]:
    """
    NEWS2 (National Early Warning Score 2).
    Only computed when enough vitals are present.
    """
    score = 0
    fields_available = 0

    # Respiratory rate
    if v.rr:
        fields_available += 1
        if v.rr <= 8:
            score += 3
        elif v.rr <= 11:
            score += 1
        elif v.rr <= 20:
            score += 0
        elif v.rr <= 24:
            score += 2
        else:
            score += 3

    # SpO2
    if v.spo2:
        fields_available += 1
        if v.spo2 <= 91:
            score += 3
        elif v.spo2 <= 93:
            score += 2
        elif v.spo2 <= 95:
            score += 1
        else:
            score += 0

    # Heart rate
    if v.hr:
        fields_available += 1
        if v.hr <= 40:
            score += 3
        elif v.hr <= 50:
            score += 1
        elif v.hr <= 90:
            score += 0
        elif v.hr <= 110:
            score += 1
        elif v.hr <= 130:
            score += 2
        else:
            score += 3

    # Systolic BP
    if v.bp:
        sys = _parse_systolic(v.bp)
        if sys:
            fields_available += 1
            if sys <= 90:
                score += 3
            elif sys <= 100:
                score += 2
            elif sys <= 110:
                score += 1
            elif sys <= 219:
                score += 0
            else:
                score += 3

    # GCS (via consciousness level proxy)
    if v.gcs:
        fields_available += 1
        if v.gcs == 15:
            score += 0      # Alert
        elif v.gcs >= 13:
            score += 3      # Confused
        else:
            score += 3      # Unresponsive/Pain

    # Temperature
    if v.temp:
        fields_available += 1
        if v.temp <= 35.0:
            score += 3
        elif v.temp <= 36.0:
            score += 1
        elif v.temp <= 38.0:
            score += 0
        elif v.temp <= 39.0:
            score += 1
        else:
            score += 2

    # Only return score if at least 3 vitals were provided
    return score if fields_available >= 3 else None

def get_immediate_actions(criticality: str, chief_complaint: Optional[str]) -> list[str]:
    """Return protocol-based immediate actions for common presentations."""
    actions_map = {
        "chest pain": [
            "Position patient semi-recumbent (45°)",
            "O2 if SpO2 < 94%",
            "12-lead ECG within 10 minutes",
            "Aspirin 324mg PO (chew) if no contraindications",
            "Large-bore IV access",
            "Continuous cardiac monitoring",
        ],
        "stroke": [
            "Note EXACT time of symptom onset",
            "Perform BEFAST assessment",
            "Keep HOB flat (0-15°)",
            "NPO — aspiration risk",
            "DO NOT give glucose unless BGL < 60",
            "Pre-notify: Stroke Alert to receiving hospital",
        ],
        "trauma": [
            "Primary survey: ABCDE",
            "Control hemorrhage — direct pressure / tourniquet",
            "C-spine precautions if indicated",
            "Large-bore IV x2, fluid resus if shocked",
            "Minimize on-scene time (Golden Hour)",
        ],
        "seizure": [
            "Protect from injury — do NOT restrain",
            "Time the seizure",
            "If > 5 min: Midazolam 10mg IM lateral thigh",
            "Position lateral recovery post-seizure",
            "Check BGL — hypoglycemia can cause seizures",
        ],
        "anaphylaxis": [
            "Epinephrine 0.3mg IM lateral thigh — IMMEDIATELY",
            "High-flow O2",
            "Large-bore IV — 1-2L NS if hypotensive",
            "Diphenhydramine 50mg IV/IM",
            "Methylprednisolone 125mg IV",
        ],
        "respiratory": [
            "Position of comfort (usually upright)",
            "High-flow O2 / CPAP if available",
            "Assess for tension pneumothorax",
            "BVM if apneic or SpO2 < 85%",
        ],
        "cardiac arrest": [
            "Confirm unresponsiveness and apnea",
            "Call for AED / defibrillator",
            "CPR: 30:2, 100-120/min, 2-2.4 inch depth",
            "Attach AED — analyze ASAP",
            "Epinephrine 1mg IV/IO every 3-5 min",
        ],
    }

    if not chief_complaint:
        return _default_actions(criticality)

    complaint_lower = chief_complaint.lower()
    for key, actions in actions_map.items():
        if key in complaint_lower:
            return actions

    return _default_actions(criticality)

def _default_actions(criticality: str) -> list[str]:
    if criticality == "CRITICAL":
        return [
            "Ensure scene safety",
            "Call for backup / ALS intercept",
            "Obtain full vital signs",
            "Establish IV access",
            "Continuous monitoring",
            "Expedite transport to appropriate facility",
        ]
    return [
        "Obtain full vital signs",
        "Conduct focused physical exam",
        "SAMPLE history",
        "Determine appropriate transport destination",
    ]