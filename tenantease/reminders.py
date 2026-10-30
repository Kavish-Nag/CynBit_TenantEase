from __future__ import annotations

from datetime import date
from typing import Any


def format_inr(value: float) -> str:
    return f"₹{value:,.0f}"


def build_reminder(charge: dict[str, Any], tone: str = "Professional", risk_label: str | None = None) -> str:
    tenant = charge.get("full_name", "Tenant")
    due = date.fromisoformat(charge["due_date"]).strftime("%d %b %Y")
    amount = format_inr(float(charge["outstanding"]))
    period = charge.get("period", "the current cycle")
    if charge["status"] == "Overdue":
        opening = {
            "Friendly": f"Hi {tenant}, hope you are doing well. This is a gentle reminder that the rent for {period} is still outstanding.",
            "Professional": f"Hello {tenant}, our records show that the rent for {period} remains outstanding after its due date.",
            "Firm": f"Hello {tenant}, the rent payment for {period} is overdue according to our records.",
        }.get(tone, f"Hello {tenant}, the rent for {period} remains outstanding.")
        deadline = "Please let us know when we can expect the payment, or contact the property manager if you have already paid."
    elif charge["status"] == "Partial":
        opening = f"Hello {tenant}, thank you for the payment received toward the {period} rent."
        deadline = "Please arrange the remaining balance or contact the property manager if you need to discuss it."
    else:
        opening = {
            "Friendly": f"Hi {tenant}, hope you are doing well. Just a quick reminder about the upcoming rent payment for {period}.",
            "Professional": f"Hello {tenant}, this is a reminder that the rent payment for {period} is approaching its due date.",
            "Firm": f"Hello {tenant}, please ensure the rent payment for {period} is completed by the due date.",
        }.get(tone, f"Hello {tenant}, this is a reminder about the upcoming rent payment.")
        deadline = "Please contact the property manager if you anticipate any difficulty making the payment on time."
    lines = [opening, "", f"Property: {charge.get('property_name', 'Property')} | Unit: {charge.get('unit_label', 'Unit')}", f"Due date: {due}", f"Outstanding balance: {amount}", "", deadline, "", "Payment details: [Insert verified payment instructions or link]", "", "Regards,", "Property Management Team"]
    if risk_label:
        lines.insert(0, f"Internal note (remove before sending): predictive payment-risk band is {risk_label}. This is an estimate, not a confirmed payment outcome.\n")
    return "\n".join(lines)
