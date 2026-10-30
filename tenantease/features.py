from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from .db import get_tenant_payment_history

FEATURES = [
    "rent_amount",
    "previous_late_rate",
    "avg_delay_days",
    "late_streak",
    "payments_observed",
    "previous_missed_count",
    "recent_delay_days",
    "due_day",
    "lease_months",
]


def _month_difference(start: date, end: date) -> int:
    return max(0, (end.year - start.year) * 12 + end.month - start.month)


def build_tenant_features(
    tenant: dict[str, Any],
    target_due_date: date,
    as_of_date: date | None = None,
    grace_days: int = 3,
) -> dict[str, float]:
    """Create leakage-conscious features using only charges before the target cycle."""
    as_of = as_of_date or date.today()
    history = get_tenant_payment_history(int(tenant["id"]), before_due_date=target_due_date)
    outcomes: list[dict[str, float | bool]] = []

    for charge in history:
        due = date.fromisoformat(charge["due_date"])
        settlement = charge["settlement_date"]
        amount_due = float(charge["amount_due"])
        amount_paid = float(charge["amount_paid"])
        if settlement is not None:
            delay = max(0, (settlement - due).days)
            late = delay > grace_days
            missed = False
        elif due <= as_of and (as_of - due).days > grace_days and amount_paid < amount_due:
            late = True
            missed = True
            delay = min(45, max(0, (as_of - due).days))
        else:
            # An unpaid but not-yet-overdue charge is not a completed outcome.
            continue
        outcomes.append({"late": late, "delay": float(delay), "missed": missed})

    # Match the model's rolling six-cycle feature window.
    recent_outcomes = outcomes[-6:]
    late_values = [bool(x["late"]) for x in recent_outcomes]
    delays = [float(x["delay"]) for x in recent_outcomes if bool(x["late"])]
    late_streak = 0
    for outcome in reversed(outcomes):
        if bool(outcome["late"]):
            late_streak += 1
        else:
            break

    last = outcomes[-1] if outcomes else None
    lease_start = date.fromisoformat(tenant["lease_start"])
    return {
        "rent_amount": float(tenant["rent_amount"]),
        "previous_late_rate": float(np.mean(late_values)) if late_values else 0.0,
        "avg_delay_days": float(np.mean(delays)) if delays else 0.0,
        "late_streak": float(late_streak),
        "payments_observed": float(len(outcomes)),
        "previous_missed_count": float(sum(bool(x["missed"]) for x in recent_outcomes)),
        "recent_delay_days": float(last["delay"]) if last else 0.0,
        "due_day": float(target_due_date.day),
        "lease_months": float(_month_difference(lease_start, target_due_date)),
    }


def explain_prediction(features: dict[str, float]) -> tuple[list[str], list[str]]:
    """Return transparent heuristic explanations and operational suggestions."""
    reasons: list[str] = []
    actions: list[str] = []
    if features["previous_late_rate"] >= 0.40:
        reasons.append(f"{features['previous_late_rate']:.0%} of observed previous cycles were late")
        actions.append("Review the tenant's recent payment pattern before the due date.")
    if features["late_streak"] >= 2:
        reasons.append(f"{int(features['late_streak'])} consecutive late cycle(s) were observed")
        actions.append("Offer a reminder several days before the due date and confirm the preferred payment channel.")
    if features["previous_missed_count"] >= 1:
        reasons.append(f"{int(features['previous_missed_count'])} previous cycle(s) remain unsettled in the recorded history")
        actions.append("Reconcile the previous outstanding balance with the payment ledger.")
    if features["avg_delay_days"] >= 5:
        reasons.append(f"Average delay on late cycles is {features['avg_delay_days']:.1f} days")
        actions.append("Schedule a courteous follow-up shortly before the usual payment date.")
    if features["payments_observed"] < 3:
        reasons.append("Limited payment history is available, so this estimate has lower confidence")
        actions.append("Treat this score as an early signal and review manually until more history is available.")
    if not reasons:
        reasons.append("No strong adverse pattern appears in the available payment history")
        actions.append("Continue standard reminders and monitor the next payment cycle.")
    return reasons, actions
