from __future__ import annotations

import calendar
import io
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tenantease.db import (
    DB_PATH,
    add_property,
    add_tenant,
    clear_all_data,
    ensure_charges_for_months,
    export_operational_data,
    get_charge_rows,
    get_collection_trend,
    get_payment_transactions,
    get_properties,
    get_tenant,
    get_tenants,
    init_db,
    record_payment,
    seed_demo_data,
    set_tenant_status,
)
from tenantease.features import FEATURES, build_tenant_features, explain_prediction
from tenantease.ml import (
    CUSTOM_MODEL_PATH,
    create_training_template,
    feature_importance_for_bundle,
    get_active_bundle,
    sample_training_dataframe,
    save_custom_bundle,
    score_features,
    train_and_evaluate,
)
from tenantease.reminders import build_reminder, format_inr


st.set_page_config(
    page_title="TenantEase | Predictive Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.025em; }
    .block-container { padding-top: 1.45rem; padding-bottom: 2.8rem; max-width: 1480px; }
    [data-testid="stSidebar"] { border-right: 1px solid rgba(100,116,139,.18); }
    .brand-mark { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.35rem; color: #075985; letter-spacing: -.04em; }
    .brand-sub { color: #64748b; font-size: .78rem; margin-top: -.25rem; }
    .eyebrow { color: #0f766e; font-size: .72rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
    .section-note { color: #64748b; font-size: .91rem; }
    .small-muted { color: #64748b; font-size: .78rem; }
    .risk-high { color: #b91c1c; font-weight: 700; }
    .risk-medium { color: #b45309; font-weight: 700; }
    .risk-low { color: #047857; font-weight: 700; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid rgba(100,116,139,.18); padding: 14px 16px; border-radius: 14px; }
    div[data-testid="stMetricLabel"] { color: #64748b; }
    div[data-testid="stMetricValue"] { font-family: 'Manrope', sans-serif; }
    .stButton button, .stDownloadButton button { border-radius: 10px; font-weight: 650; }
    div[data-testid="stForm"] { border: 1px solid rgba(100,116,139,.18); border-radius: 14px; padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the demonstration model...")
def load_bundle_cached(custom_model_mtime: float) -> dict[str, Any]:
    return get_active_bundle()


def month_period(offset: int = 0) -> str:
    today = date.today()
    absolute = today.year * 12 + (today.month - 1) + offset
    return f"{absolute // 12:04d}-{absolute % 12 + 1:02d}"


def month_label(period: str) -> str:
    year, month = map(int, period.split("-"))
    return date(year, month, 1).strftime("%b %Y")


def get_bundle() -> dict[str, Any]:
    mtime = CUSTOM_MODEL_PATH.stat().st_mtime if CUSTOM_MODEL_PATH.exists() else 0.0
    return load_bundle_cached(mtime)


def make_predictions(bundle: dict[str, Any]) -> pd.DataFrame:
    tenants = get_tenants(active_only=True)
    next_period = month_period(1)
    charges = [c for c in get_charge_rows() if c["period"] == next_period]
    charge_lookup = {c["tenant_id"]: c for c in charges}
    rows = []
    feature_rows = []
    contexts = []
    for tenant in tenants:
        charge = charge_lookup.get(tenant["id"])
        if not charge:
            continue
        target_due = date.fromisoformat(charge["due_date"])
        features = build_tenant_features(tenant, target_due)
        feature_rows.append(features)
        contexts.append((tenant, charge, features))
    probabilities = score_features(bundle, feature_rows)
    for probability, (tenant, charge, features) in zip(probabilities, contexts):
        probability = float(probability)
        risk = "High" if probability >= 0.65 else "Medium" if probability >= 0.35 else "Low"
        coverage = min(6, int(features["payments_observed"]))
        reasons, actions = explain_prediction(features)
        rows.append(
            {
                "tenant_id": tenant["id"],
                "tenant": tenant["full_name"],
                "property": tenant["property_name"],
                "unit": tenant["unit_label"],
                "due_date": charge["due_date"],
                "rent": float(charge["amount_due"]),
                "late_probability": probability,
                "risk": risk,
                "history_cycles": int(features["payments_observed"]),
                "history_coverage": f"{coverage}/6 cycles",
                "reasons": reasons,
                "actions": actions,
                "features": features,
                "charge": charge,
            }
        )
    return pd.DataFrame(rows).sort_values("late_probability", ascending=False) if rows else pd.DataFrame()


def show_header(title: str, subtitle: str) -> None:
    st.markdown(f"<div class='eyebrow'>TenantEase / Predictive Intelligence</div>", unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<div class='section-note'>{subtitle}</div>", unsafe_allow_html=True)


def get_current_month_charges() -> list[dict[str, Any]]:
    return get_charge_rows(period=month_period(0))


def dashboard_page(bundle: dict[str, Any]) -> None:
    show_header("Property operations, made clearer.", "A single workspace for occupancy, rent collection, tenant records and predictive follow-up.")
    current = get_current_month_charges()
    properties = get_properties()
    tenants = get_tenants(active_only=True)
    due_total = sum(float(c["amount_due"]) for c in current)
    collected = sum(float(c["amount_paid"]) for c in current)
    outstanding = sum(float(c["outstanding"]) for c in current)
    overdue = sum(float(c["outstanding"]) for c in current if c["status"] == "Overdue")
    occupancy = sum(int(p["active_tenants"] or 0) for p in properties)
    capacity = sum(int(p["total_units"]) for p in properties)
    predictions = make_predictions(bundle)
    at_risk = int((predictions["risk"] == "High").sum()) if not predictions.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active properties", f"{len(properties):,}")
    c2.metric("Occupied units", f"{occupancy:,}/{capacity:,}", f"{(occupancy / capacity * 100):.0f}% occupancy" if capacity else "No units configured")
    c3.metric("Collected this month", format_inr(collected), f"{(collected / due_total * 100):.0f}% of billed" if due_total else "No rent billed")
    c4.metric("Outstanding this month", format_inr(outstanding), f"{format_inr(overdue)} overdue")
    c5.metric("High-risk next cycle", f"{at_risk:,}", f"{month_label(month_period(1))} forecast")

    st.write("")
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Collection trend")
        trend = pd.DataFrame(get_collection_trend(6))
        if not trend.empty:
            trend["month"] = trend["period"].apply(month_label)
            long = trend.melt(id_vars="month", value_vars=["due", "collected"], var_name="type", value_name="amount")
            long["type"] = long["type"].map({"due": "Billed", "collected": "Collected"})
            fig = px.bar(long, x="month", y="amount", color="type", barmode="group", text_auto=".2s")
            fig.update_layout(height=330, margin=dict(l=0, r=0, t=12, b=0), legend_title_text="", yaxis_title="Amount (₹)", xaxis_title="", plot_bgcolor="white", paper_bgcolor="white")
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Collection history will appear here after rent cycles are recorded.")
    with right:
        st.subheader("Current month status")
        counts = pd.Series([c["status"] for c in current]).value_counts().reindex(["Paid", "Partial", "Pending", "Overdue"], fill_value=0)
        if counts.sum():
            fig = px.pie(names=counts.index, values=counts.values, hole=0.68)
            fig.update_layout(height=330, margin=dict(l=0, r=0, t=12, b=0), legend_title_text="", showlegend=True, paper_bgcolor="white")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No current-month charges found.")

    st.subheader("Priority follow-ups")
    if not predictions.empty:
        view = predictions[["tenant", "property", "unit", "due_date", "late_probability", "risk", "history_coverage"]].head(6).copy()
        view["late_probability"] = view["late_probability"].map(lambda x: f"{x:.0%}")
        view.columns = ["Tenant", "Property", "Unit", "Next due date", "Estimated late risk", "Risk band", "History coverage"]
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.caption("Predictions are estimates from the currently selected model. Review the history coverage before acting.")
    else:
        st.info("Add active tenants and rent cycles to generate predictions.")


def properties_tenants_page() -> None:
    show_header("Properties & tenants", "Maintain the property portfolio and the tenant records used by rent collection and risk scoring.")
    tab1, tab2, tab3 = st.tabs(["Property portfolio", "Tenant directory", "Add records"])
    with tab1:
        properties = get_properties()
        if properties:
            cols = st.columns(3)
            for idx, prop in enumerate(properties):
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"### {prop['name']}")
                        st.caption(prop["address"] or "Address not provided")
                        active = int(prop["active_tenants"] or 0)
                        total = int(prop["total_units"])
                        st.metric("Occupied units", f"{active}/{total}", f"{(active / total * 100):.0f}% occupied" if total else "")
                        st.progress(min(1.0, active / total) if total else 0.0)
                        st.caption(f"{max(0, total-active)} unit(s) available")
        else:
            st.info("No properties yet. Add a property from the Add records tab.")
    with tab2:
        tenants = get_tenants()
        if tenants:
            df = pd.DataFrame(tenants)[["full_name", "property_name", "unit_label", "rent_amount", "rent_due_day", "lease_start", "lease_end", "status", "email", "phone"]]
            df = df.rename(columns={"full_name": "Tenant", "property_name": "Property", "unit_label": "Unit", "rent_amount": "Monthly rent (₹)", "rent_due_day": "Due day", "lease_start": "Lease start", "lease_end": "Lease end", "status": "Status", "email": "Email", "phone": "Phone"})
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.divider()
            with st.expander("Change a tenant's active status"):
                tenant_lookup = {f"{t['full_name']} · {t['property_name']} · {t['unit_label']} (#{t['id']})": t for t in tenants}
                selected_label = st.selectbox("Tenant", list(tenant_lookup), key="status_tenant")
                selected = tenant_lookup[selected_label]
                new_status = st.selectbox("New status", ["Active", "Inactive"], index=0 if selected["status"] == "Active" else 1)
                if st.button("Update tenant status", key="update_status"):
                    set_tenant_status(selected["id"], new_status)
                    st.success("Tenant status updated.")
                    st.rerun()
        else:
            st.info("No tenant records yet.")
    with tab3:
        left, right = st.columns(2)
        with left:
            st.subheader("Add property")
            with st.form("add_property_form", clear_on_submit=True):
                name = st.text_input("Property name", placeholder="e.g. Lakeview Apartments")
                address = st.text_input("Address / locality", placeholder="e.g. Baner, Pune")
                units = st.number_input("Total units", min_value=1, max_value=10000, value=20, step=1)
                submitted = st.form_submit_button("Add property", use_container_width=True)
                if submitted:
                    if not name.strip():
                        st.error("Enter a property name.")
                    else:
                        add_property(name, address, int(units))
                        st.success("Property added.")
                        st.rerun()
        with right:
            st.subheader("Add tenant")
            props = get_properties()
            if not props:
                st.info("Add a property before adding its tenants.")
            else:
                prop_map = {f"{p['name']} · {p['address']} (#{p['id']})": p for p in props}
                with st.form("add_tenant_form", clear_on_submit=True):
                    prop_label = st.selectbox("Property", list(prop_map))
                    unit_label = st.text_input("Unit / room", placeholder="e.g. B-204")
                    full_name = st.text_input("Tenant name")
                    email = st.text_input("Email", placeholder="Optional")
                    phone = st.text_input("Phone", placeholder="Optional")
                    rent = st.number_input("Monthly rent (₹)", min_value=0.0, value=18000.0, step=500.0)
                    due_day = st.number_input("Rent due day", min_value=1, max_value=28, value=5)
                    lease_start = st.date_input("Lease start", value=date.today())
                    lease_end_enabled = st.checkbox("Set a lease end date")
                    lease_end = st.date_input("Lease end", value=date.today() + timedelta(days=365), disabled=not lease_end_enabled)
                    submitted = st.form_submit_button("Add tenant", use_container_width=True)
                    if submitted:
                        selected_prop = prop_map[prop_label]
                        occupied = {t["unit_label"].lower() for t in get_tenants(active_only=True) if t["property_id"] == selected_prop["id"]}
                        active_in_property = sum(1 for t in get_tenants(active_only=True) if t["property_id"] == selected_prop["id"])
                        if not full_name.strip() or not unit_label.strip():
                            st.error("Tenant name and unit are required.")
                        elif active_in_property >= int(selected_prop["total_units"]):
                            st.error("This property has reached its configured unit capacity.")
                        elif unit_label.strip().lower() in occupied:
                            st.error("That unit is already assigned to an active tenant.")
                        elif lease_end_enabled and lease_end < lease_start:
                            st.error("Lease end cannot be earlier than lease start.")
                        else:
                            try:
                                tenant_id = add_tenant(selected_prop["id"], unit_label, full_name, email, phone, rent, int(due_day), lease_start, lease_end if lease_end_enabled else None)
                                ensure_charges_for_months(2)
                                st.success(f"Tenant added (ID {tenant_id}). Current and next-cycle charges have been created. Historical records are not fabricated.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not add tenant: {exc}")


def payments_page() -> None:
    show_header("Rent & payments", "Record payments against monthly rent charges and keep the collection ledger up to date.")
    all_charges = get_charge_rows()
    if not all_charges:
        st.info("No rent charges are available yet.")
        return
    total_outstanding = sum(c["outstanding"] for c in all_charges if c["due_date"] <= date.today().isoformat())
    total_overdue = sum(c["outstanding"] for c in all_charges if c["status"] == "Overdue")
    a, b, c = st.columns(3)
    a.metric("Outstanding due", format_inr(total_outstanding))
    b.metric("Overdue balance", format_inr(total_overdue))
    c.metric("Payment transactions", f"{len(get_payment_transactions(10000)):,}")

    tab1, tab2 = st.tabs(["Charge ledger", "Record payment"])
    with tab1:
        statuses = ["All", "Paid", "Partial", "Pending", "Overdue"]
        selected_status = st.selectbox("Filter by status", statuses)
        period_values = sorted({c["period"] for c in all_charges}, reverse=True)
        selected_period = st.selectbox("Billing period", ["All periods"] + period_values, index=0)
        filtered = all_charges
        if selected_status != "All":
            filtered = [c for c in filtered if c["status"] == selected_status]
        if selected_period != "All periods":
            filtered = [c for c in filtered if c["period"] == selected_period]
        df = pd.DataFrame(filtered)
        if not df.empty:
            display = df[["full_name", "property_name", "unit_label", "period", "due_date", "amount_due", "amount_paid", "outstanding", "status", "days_overdue"]].copy()
            display.columns = ["Tenant", "Property", "Unit", "Period", "Due date", "Billed (₹)", "Paid (₹)", "Outstanding (₹)", "Status", "Days overdue"]
            st.dataframe(display, hide_index=True, use_container_width=True)
        else:
            st.info("No charges match the selected filters.")
    with tab2:
        open_charges = [c for c in all_charges if c["outstanding"] > 0.009]
        if not open_charges:
            st.success("All generated rent charges are fully paid.")
        else:
            charge_map = {
                f"{c['full_name']} · {c['property_name']} / {c['unit_label']} · {c['period']} · Due {c['due_date']} · Balance {format_inr(c['outstanding'])} · #{c['charge_id']}": c
                for c in open_charges
            }
            with st.form("record_payment_form"):
                selected_label = st.selectbox("Select rent charge", list(charge_map))
                selected = charge_map[selected_label]
                st.caption(f"Billed: {format_inr(selected['amount_due'])} | Already paid: {format_inr(selected['amount_paid'])} | Remaining: {format_inr(selected['outstanding'])}")
                amount = st.number_input("Payment amount (₹)", min_value=0.01, max_value=float(max(0.01, selected["outstanding"])), value=float(max(0.01, selected["outstanding"])), step=500.0)
                paid_on = st.date_input("Payment date", value=date.today(), max_value=date.today())
                method = st.selectbox("Payment method", ["UPI", "Bank transfer", "NEFT", "Cash", "Cheque", "Other"])
                reference = st.text_input("Transaction reference", placeholder="Optional UTR / reference number")
                submitted = st.form_submit_button("Record payment", use_container_width=True)
                if submitted:
                    try:
                        record_payment(selected["charge_id"], paid_on, amount, method, reference)
                        st.success("Payment recorded. The ledger and predictive features now use this transaction.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
    st.divider()
    st.subheader("Recent payment transactions")
    transactions = get_payment_transactions(12)
    if transactions:
        tx = pd.DataFrame(transactions)
        tx = tx[["paid_on", "full_name", "property_name", "unit_label", "period", "amount", "method", "reference"]]
        tx.columns = ["Paid on", "Tenant", "Property", "Unit", "Period", "Amount (₹)", "Method", "Reference"]
        st.dataframe(tx, hide_index=True, use_container_width=True)
    else:
        st.caption("Payment transactions will appear here after the first payment is recorded.")


def predictive_page(bundle: dict[str, Any]) -> None:
    show_header("Predictive intelligence", "Estimate which upcoming rent payments may be delayed, understand the signals, and plan earlier follow-ups.")
    predictions = make_predictions(bundle)
    target_period = month_period(1)
    st.info(
        f"Forecast cycle: **{month_label(target_period)}**. The demo model is trained on synthetic payment histories unless you upload and activate a labelled historical dataset in Data & Model. Scores are decision-support estimates, not facts about a tenant."
    )
    if predictions.empty:
        st.warning("No upcoming rent charges are available. Add an active tenant or verify the lease dates.")
        return
    high = int((predictions["risk"] == "High").sum())
    medium = int((predictions["risk"] == "Medium").sum())
    low = int((predictions["risk"] == "Low").sum())
    expected = float((predictions["rent"] * predictions["late_probability"]).sum())
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Tenants forecast", f"{len(predictions):,}")
    x2.metric("High risk", f"{high:,}")
    x3.metric("Medium risk", f"{medium:,}")
    x4.metric("Risk-weighted rent exposure", format_inr(expected))

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Predicted late-payment probability")
        chart = predictions.head(15).copy()
        chart["tenant_unit"] = chart["tenant"] + " · " + chart["unit"]
        chart = chart.sort_values("late_probability", ascending=True)
        fig = px.bar(chart, x="late_probability", y="tenant_unit", color="risk", orientation="h", text=chart["late_probability"].map(lambda x: f"{x:.0%}"))
        fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Estimated probability")
        fig.update_yaxes(title="")
        fig.update_layout(height=440, margin=dict(l=0, r=8, t=8, b=0), legend_title_text="Risk band", plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Risk distribution")
        band_counts = predictions["risk"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
        fig = px.pie(names=band_counts.index, values=band_counts.values, hole=0.66)
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=8, b=0), showlegend=True, legend_title_text="", paper_bgcolor="white")
        fig.update_traces(textinfo="value+label", textposition="inside")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Model selected: **{bundle['model_name']}** | Training source: {bundle['source']} | Rows: {bundle.get('rows', 0):,}")

    st.subheader("Tenant-level prediction review")
    display = predictions[["tenant", "property", "unit", "due_date", "rent", "late_probability", "risk", "history_coverage"]].copy()
    display["rent"] = display["rent"].map(format_inr)
    display["late_probability"] = display["late_probability"].map(lambda x: f"{x:.1%}")
    display.columns = ["Tenant", "Property", "Unit", "Due date", "Monthly rent", "Late probability", "Risk band", "History coverage"]
    st.dataframe(display, hide_index=True, use_container_width=True)

    st.subheader("Explain a prediction")
    pred_lookup = {f"{r['tenant']} · {r['property']} / {r['unit']} · {r['risk']} ({r['late_probability']:.0%})": r for _, r in predictions.iterrows()}
    label = st.selectbox("Choose a tenant", list(pred_lookup), key="prediction_detail")
    selected = pred_lookup[label]
    detail_left, detail_right = st.columns([1, 1])
    with detail_left:
        st.metric("Estimated late-payment probability", f"{selected['late_probability']:.1%}", selected["risk"] + " risk")
        st.markdown("**Signals observed in recorded history**")
        for reason in selected["reasons"]:
            st.markdown(f"- {reason}")
        st.markdown("**Suggested property-manager actions**")
        for action in selected["actions"]:
            st.markdown(f"- {action}")
    with detail_right:
        st.markdown("**Input feature snapshot**")
        feature_names = {
            "previous_late_rate": "Previous late rate",
            "avg_delay_days": "Average delay days",
            "late_streak": "Consecutive late cycles",
            "payments_observed": "Observed completed cycles",
            "previous_missed_count": "Unsettled previous cycles",
            "recent_delay_days": "Most recent delay days",
            "due_day": "Rent due day",
            "lease_months": "Lease age (months)",
            "rent_amount": "Monthly rent (₹)",
        }
        rows = []
        for key, label_text in feature_names.items():
            value = selected["features"][key]
            if key == "previous_late_rate":
                shown = f"{value:.0%}"
            elif key == "rent_amount":
                shown = format_inr(value)
            else:
                shown = f"{value:.1f}"
            rows.append({"Feature": label_text, "Value": shown})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Risk bands are demo thresholds: Low < 35%, Medium 35% to 64.9%, High ≥ 65%. They should be calibrated on TenantEase's real historical data before production use.")


def reminders_page(bundle: dict[str, Any]) -> None:
    show_header("Smart follow-ups", "Prepare respectful, context-aware rent reminders from the actual charge ledger. Messages are drafts only and are never sent automatically.")
    open_charges = [c for c in get_charge_rows() if c["outstanding"] > 0.009]
    if not open_charges:
        st.success("There are no outstanding rent charges.")
        return
    predictions = make_predictions(bundle)
    pred_by_tenant = {int(r["tenant_id"]): r for _, r in predictions.iterrows()} if not predictions.empty else {}
    options = {
        f"{c['full_name']} · {c['property_name']} / {c['unit_label']} · {c['period']} · {c['status']} · {format_inr(c['outstanding'])} · #{c['charge_id']}": c
        for c in open_charges
    }
    selected_label = st.selectbox("Choose an outstanding charge", list(options))
    charge = options[selected_label]
    p = pred_by_tenant.get(int(charge["tenant_id"]))
    risk_label = p["risk"] if p is not None else None
    tone = st.radio("Message tone", ["Friendly", "Professional", "Firm"], horizontal=True, index=1)
    if p is not None:
        st.caption(f"Next-cycle predictive band for this tenant: {risk_label}. This is an internal estimate and should not be included in the tenant-facing message.")
    draft = build_reminder(charge, tone=tone)
    st.markdown("**Editable reminder draft**")
    edited = st.text_area("Message", value=draft, height=300, label_visibility="collapsed")
    st.download_button("Download reminder (.txt)", data=edited.encode("utf-8"), file_name=f"rent_reminder_charge_{charge['charge_id']}.txt", mime="text/plain", use_container_width=True)
    st.caption("Before sending, verify the tenant, amount, due date and payment instructions. Sending integrations are intentionally not enabled in this prototype.")


def data_model_page() -> None:
    show_header("Data & model controls", "Inspect model quality, export operational data, and replace synthetic training with a labelled historical dataset when available.")
    st.subheader("Active model")
    bundle = get_bundle()
    st.write(f"**Model:** {bundle['model_name']}  |  **Training source:** {bundle['source']}  |  **Rows:** {bundle.get('rows', 0):,}  |  **Late outcome rate:** {bundle.get('positive_rate', 0):.1%}")
    metrics_by_model = bundle.get("metrics", {})
    if metrics_by_model:
        metric_rows = []
        for name, metrics in metrics_by_model.items():
            metric_rows.append({"Model": name, "Accuracy": metrics["accuracy"], "Precision": metrics["precision"], "Recall": metrics["recall"], "PR-AUC": metrics["pr_auc"], "ROC-AUC": metrics["roc_auc"], "Brier score": metrics["brier"]})
        st.dataframe(pd.DataFrame(metric_rows).style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "PR-AUC", "ROC-AUC", "Brier score"]}), hide_index=True, use_container_width=True)
        st.caption("Holdout metrics are calculated using a tenant-group split for the built-in demo data. They describe synthetic data performance, not real TenantEase production performance.")

    st.subheader("Train on labelled historical data")
    st.markdown("Upload a CSV containing the feature columns below and a `late_next_cycle` target (0 = on time, 1 = late). Include `tenant_group` where possible so rows from the same tenant stay together during evaluation.")
    st.code(", ".join(FEATURES + ["late_next_cycle", "tenant_group"]), language="text")
    st.download_button("Download sample training CSV", data=create_training_template(), file_name="tenantease_training_template_synthetic.csv", mime="text/csv")
    upload = st.file_uploader("Upload labelled training CSV", type=["csv"], key="training_csv")
    if upload is not None:
        try:
            train_df = pd.read_csv(upload)
            st.caption(f"Loaded {len(train_df):,} row(s) and {len(train_df.columns)} column(s). The file is not uploaded anywhere; it is processed locally by this app.")
            st.dataframe(train_df.head(8), use_container_width=True)
            if st.button("Train, evaluate and activate custom model", type="primary"):
                with st.spinner("Training and evaluating candidate models..."):
                    custom_bundle = train_and_evaluate(train_df, group_col="tenant_group")
                    save_custom_bundle(custom_bundle)
                    load_bundle_cached.clear()
                st.success(f"Activated {custom_bundle['model_name']} trained on {custom_bundle['rows']:,} labelled rows.")
                st.rerun()
        except Exception as exc:
            st.error(f"Could not read the training file: {exc}")

    if bundle["source"] == "uploaded labelled dataset":
        if st.button("Switch back to synthetic demo model"):
            if CUSTOM_MODEL_PATH.exists():
                CUSTOM_MODEL_PATH.unlink()
            load_bundle_cached.clear()
            st.success("Synthetic demonstration model restored.")
            st.rerun()

    st.divider()
    st.subheader("Operational data")
    exports = export_operational_data()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in exports.items():
            archive.writestr(name, content)
    st.download_button("Export properties, tenants, charges and payments (.zip)", data=zip_buffer.getvalue(), file_name="tenantease_operational_data.zip", mime="application/zip")
    st.caption(f"Local SQLite database: `{DB_PATH}`. Back up this file before moving the app or changing environments.")

    with st.expander("Reset demo dataset"):
        st.warning("This permanently deletes all local properties, tenants, rent charges and payment records, then recreates the synthetic sample. Export your data first. A custom uploaded model file is not deleted.")
        confirm = st.checkbox("I understand this will replace the current operational database.", key="reset_confirm")
        if st.button("Reset database and reseed demo records", disabled=not confirm, type="secondary"):
            clear_all_data()
            seed_demo_data(force=False)
            st.success("Demo data restored.")
            st.rerun()


def main() -> None:
    init_db()
    seed_demo_data()
    ensure_charges_for_months(2)

    with st.sidebar:
        st.markdown("<div class='brand-mark'>TenantEase</div><div class='brand-sub'>Predictive Intelligence</div>", unsafe_allow_html=True)
        st.write("")
        page = st.radio(
            "Workspace",
            ["Overview", "Properties & Tenants", "Rent & Payments", "Predictive Intelligence", "Smart Follow-ups", "Data & Model"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Demo workspace**")
        st.caption("Local SQLite · Synthetic sample records")
        st.caption("No real messages are sent. No external API key required.")
        st.divider()
        st.markdown("<span class='small-muted'>TenantEase extension prototype · v1.0</span>", unsafe_allow_html=True)

    bundle = get_bundle()
    if page == "Overview":
        dashboard_page(bundle)
    elif page == "Properties & Tenants":
        properties_tenants_page()
    elif page == "Rent & Payments":
        payments_page()
    elif page == "Predictive Intelligence":
        predictive_page(bundle)
    elif page == "Smart Follow-ups":
        reminders_page(bundle)
    elif page == "Data & Model":
        data_model_page()


if __name__ == "__main__":
    main()
