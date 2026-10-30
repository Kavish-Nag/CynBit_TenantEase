from __future__ import annotations

import calendar
import os
import random
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "tenantease.db"
DB_PATH = Path(os.getenv("TENANTEASE_DB_PATH", str(DEFAULT_DB)))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_session():
    """Connection context that commits/rolls back and always closes SQLite."""
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db_session() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                address TEXT NOT NULL DEFAULT '',
                total_units INTEGER NOT NULL CHECK(total_units > 0),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS tenants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_id INTEGER NOT NULL REFERENCES properties(id),
                unit_label TEXT NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL DEFAULT '',
                phone TEXT NOT NULL DEFAULT '',
                rent_amount REAL NOT NULL CHECK(rent_amount >= 0),
                rent_due_day INTEGER NOT NULL DEFAULT 5 CHECK(rent_due_day BETWEEN 1 AND 28),
                lease_start TEXT NOT NULL,
                lease_end TEXT,
                status TEXT NOT NULL DEFAULT 'Active' CHECK(status IN ('Active','Inactive')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(property_id, unit_label)
            );
            CREATE TABLE IF NOT EXISTS rent_charges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id INTEGER NOT NULL REFERENCES tenants(id),
                period TEXT NOT NULL,
                due_date TEXT NOT NULL,
                amount_due REAL NOT NULL CHECK(amount_due >= 0),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tenant_id, period)
            );
            CREATE TABLE IF NOT EXISTS payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                charge_id INTEGER NOT NULL REFERENCES rent_charges(id),
                paid_on TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                method TEXT NOT NULL DEFAULT 'Bank transfer',
                reference TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_tenants_property ON tenants(property_id);
            CREATE INDEX IF NOT EXISTS idx_charges_tenant_due ON rent_charges(tenant_id, due_date);
            CREATE INDEX IF NOT EXISTS idx_payments_charge ON payments(charge_id);
            """
        )


def _month_shift(year: int, month: int, offset: int) -> tuple[int, int]:
    absolute = year * 12 + (month - 1) + offset
    return absolute // 12, absolute % 12 + 1


def _due_date(period: str, due_day: int) -> date:
    year, month = map(int, period.split("-"))
    day = min(max(1, int(due_day)), calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _month_period(dt: date) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def add_property(name: str, address: str, total_units: int) -> int:
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO properties(name,address,total_units) VALUES(?,?,?)",
            (name.strip(), address.strip(), int(total_units)),
        )
        return int(cur.lastrowid)


def add_tenant(
    property_id: int,
    unit_label: str,
    full_name: str,
    email: str,
    phone: str,
    rent_amount: float,
    rent_due_day: int,
    lease_start: date,
    lease_end: date | None = None,
) -> int:
    with db_session() as conn:
        cur = conn.execute(
            """INSERT INTO tenants
            (property_id,unit_label,full_name,email,phone,rent_amount,rent_due_day,lease_start,lease_end)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                int(property_id), unit_label.strip(), full_name.strip(), email.strip(), phone.strip(),
                float(rent_amount), int(rent_due_day), lease_start.isoformat(),
                lease_end.isoformat() if lease_end else None,
            ),
        )
        return int(cur.lastrowid)


def set_tenant_status(tenant_id: int, status: str) -> None:
    if status not in {"Active", "Inactive"}:
        raise ValueError("Status must be Active or Inactive")
    with db_session() as conn:
        conn.execute("UPDATE tenants SET status=? WHERE id=?", (status, int(tenant_id)))


def get_properties() -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute(
            """SELECT p.*, COUNT(t.id) AS occupied_units,
            SUM(CASE WHEN t.status='Active' THEN 1 ELSE 0 END) AS active_tenants
            FROM properties p LEFT JOIN tenants t ON t.property_id=p.id
            GROUP BY p.id ORDER BY p.name"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_tenants(active_only: bool = False) -> list[dict[str, Any]]:
    sql = """SELECT t.*, p.name AS property_name, p.address AS property_address
             FROM tenants t JOIN properties p ON p.id=t.property_id"""
    if active_only:
        sql += " WHERE t.status='Active'"
    sql += " ORDER BY p.name, t.unit_label"
    with db_session() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def get_tenant(tenant_id: int) -> dict[str, Any] | None:
    with db_session() as conn:
        row = conn.execute(
            """SELECT t.*, p.name AS property_name FROM tenants t
               JOIN properties p ON p.id=t.property_id WHERE t.id=?""",
            (int(tenant_id),),
        ).fetchone()
    return dict(row) if row else None


def ensure_charges_for_months(months: int = 2) -> int:
    """Create current and future charges without fabricating historical payment records."""
    today = date.today()
    created = 0
    tenants = get_tenants(active_only=True)
    with db_session() as conn:
        for tenant in tenants:
            lease_start = date.fromisoformat(tenant["lease_start"])
            lease_end = date.fromisoformat(tenant["lease_end"]) if tenant["lease_end"] else None
            for offset in range(months):
                year, month = _month_shift(today.year, today.month, offset)
                due = _due_date(f"{year:04d}-{month:02d}", tenant["rent_due_day"])
                if lease_start > due or (lease_end and lease_end < due):
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO rent_charges(tenant_id,period,due_date,amount_due) VALUES(?,?,?,?)",
                    (tenant["id"], f"{year:04d}-{month:02d}", due.isoformat(), tenant["rent_amount"]),
                )
                created += cur.rowcount
    return created


def get_charge_rows(period: str | None = None, include_future: bool = True) -> list[dict[str, Any]]:
    sql = """SELECT c.id AS charge_id, c.tenant_id, c.period, c.due_date, c.amount_due,
                    t.full_name, t.unit_label, t.email, t.phone, t.rent_amount,
                    p.name AS property_name,
                    COALESCE(SUM(pay.amount),0) AS amount_paid,
                    MAX(pay.paid_on) AS last_payment_date
             FROM rent_charges c
             JOIN tenants t ON t.id=c.tenant_id
             JOIN properties p ON p.id=t.property_id
             LEFT JOIN payments pay ON pay.charge_id=c.id"""
    params: list[Any] = []
    clauses = []
    if period:
        clauses.append("c.period=?")
        params.append(period)
    if not include_future:
        clauses.append("c.due_date<=?")
        params.append(date.today().isoformat())
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " GROUP BY c.id ORDER BY c.due_date DESC, p.name, t.unit_label"
    with db_session() as conn:
        rows = conn.execute(sql, params).fetchall()
    today = date.today()
    result = []
    for raw in rows:
        row = dict(raw)
        row["outstanding"] = max(0.0, round(float(row["amount_due"]) - float(row["amount_paid"]), 2))
        due = date.fromisoformat(row["due_date"])
        if row["outstanding"] <= 0.009:
            row["status"] = "Paid"
        elif float(row["amount_paid"]) > 0:
            row["status"] = "Partial"
        elif due < today:
            row["status"] = "Overdue"
        else:
            row["status"] = "Pending"
        row["days_overdue"] = max(0, (today - due).days) if row["status"] == "Overdue" else 0
        result.append(row)
    return result


def record_payment(charge_id: int, paid_on: date, amount: float, method: str, reference: str = "") -> int:
    if amount <= 0:
        raise ValueError("Payment amount must be greater than zero")
    charge = next((c for c in get_charge_rows() if c["charge_id"] == int(charge_id)), None)
    if not charge:
        raise ValueError("Charge was not found")
    if amount > charge["outstanding"] + 0.009:
        raise ValueError("Payment cannot exceed the remaining outstanding amount")
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO payments(charge_id,paid_on,amount,method,reference) VALUES(?,?,?,?,?)",
            (int(charge_id), paid_on.isoformat(), float(amount), method, reference.strip()),
        )
        return int(cur.lastrowid)


def get_tenant_payment_history(tenant_id: int, before_due_date: date | None = None) -> list[dict[str, Any]]:
    sql = """SELECT c.id AS charge_id,c.period,c.due_date,c.amount_due,
                    COALESCE(SUM(p.amount),0) AS amount_paid
             FROM rent_charges c LEFT JOIN payments p ON p.charge_id=c.id
             WHERE c.tenant_id=?"""
    params: list[Any] = [int(tenant_id)]
    if before_due_date:
        sql += " AND c.due_date < ?"
        params.append(before_due_date.isoformat())
    sql += " GROUP BY c.id ORDER BY c.due_date"
    with db_session() as conn:
        rows = conn.execute(sql, params).fetchall()
        history = []
        for row in rows:
            data = dict(row)
            payments = conn.execute(
                "SELECT paid_on,amount FROM payments WHERE charge_id=? ORDER BY paid_on,id",
                (data["charge_id"],),
            ).fetchall()
            cumulative = 0.0
            settlement = None
            for pay in payments:
                cumulative += float(pay["amount"])
                if cumulative + 0.009 >= float(data["amount_due"]):
                    settlement = date.fromisoformat(pay["paid_on"])
                    break
            data["settlement_date"] = settlement
            data["payment_count"] = len(payments)
            history.append(data)
    return history


def get_collection_trend(month_count: int = 6) -> list[dict[str, Any]]:
    today = date.today()
    periods = []
    for offset in range(month_count - 1, -1, -1):
        year, month = _month_shift(today.year, today.month, -offset)
        periods.append(f"{year:04d}-{month:02d}")
    charges = get_charge_rows()
    result = []
    for period in periods:
        selected = [c for c in charges if c["period"] == period]
        due = sum(float(c["amount_due"]) for c in selected)
        paid = sum(float(c["amount_paid"]) for c in selected)
        result.append({"period": period, "due": due, "collected": paid})
    return result


def get_payment_transactions(limit: int = 20) -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute(
            """SELECT p.id,p.paid_on,p.amount,p.method,p.reference,t.full_name,t.unit_label,
                      prop.name AS property_name,c.period
               FROM payments p JOIN rent_charges c ON c.id=p.charge_id
               JOIN tenants t ON t.id=c.tenant_id JOIN properties prop ON prop.id=t.property_id
               ORDER BY p.paid_on DESC,p.id DESC LIMIT ?""",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def export_operational_data() -> dict[str, bytes]:
    import csv
    import io

    tables = {
        "properties.csv": ("SELECT id,name,address,total_units,created_at FROM properties",),
        "tenants.csv": ("SELECT * FROM tenants",),
        "rent_charges.csv": ("SELECT * FROM rent_charges",),
        "payments.csv": ("SELECT * FROM payments",),
    }
    output: dict[str, bytes] = {}
    with db_session() as conn:
        for filename, (query,) in tables.items():
            rows = conn.execute(query).fetchall()
            buffer = io.StringIO()
            if rows:
                writer = csv.writer(buffer)
                writer.writerow(rows[0].keys())
                writer.writerows([tuple(r) for r in rows])
            output[filename] = buffer.getvalue().encode("utf-8")
    return output


def clear_all_data() -> None:
    with db_session() as conn:
        conn.execute("DELETE FROM payments")
        conn.execute("DELETE FROM rent_charges")
        conn.execute("DELETE FROM tenants")
        conn.execute("DELETE FROM properties")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('properties','tenants','rent_charges','payments')")


def seed_demo_data(force: bool = False) -> None:
    """Seed transparent, synthetic demo records on first launch."""
    init_db()
    with db_session() as conn:
        count = conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
    if count and not force:
        return
    if force:
        clear_all_data()

    rng = random.Random(42)
    properties = [
        ("Maple Residency", "Baner, Pune", 24),
        ("Riverstone Apartments", "Wakad, Pune", 18),
        ("Cedar Heights", "Hinjawadi, Pune", 16),
    ]
    prop_ids = [add_property(*item) for item in properties]
    first_names = ["Aarav", "Isha", "Kabir", "Meera", "Rohan", "Ananya", "Vihaan", "Diya", "Arjun", "Sara", "Neil", "Tara", "Dev", "Mira", "Aditya", "Kiara", "Ishaan", "Naina"]
    last_names = ["Shah", "Kulkarni", "Mehta", "Patil", "Rao", "Kapoor", "Joshi", "Desai", "Nair", "Malhotra", "Singh", "Iyer"]
    today = date.today()
    start_month = _month_shift(today.year, today.month, -11)
    due_periods = []
    for i in range(12):
        y, m = _month_shift(start_month[0], start_month[1], i)
        due_periods.append(f"{y:04d}-{m:02d}")

    tenant_specs = []
    for idx in range(36):
        prop_id = prop_ids[idx % len(prop_ids)]
        prop_idx = idx % len(prop_ids)
        unit = f"{chr(65 + (idx // 12))}-{101 + (idx % 12):03d}"
        name = f"{rng.choice(first_names)} {rng.choice(last_names)}"
        risk_band = rng.choices(["low", "medium", "high"], weights=[0.48, 0.32, 0.20])[0]
        rent = rng.randrange(11000, 36001, 500)
        due_day = rng.choice([1, 3, 5, 7, 10])
        lease_start = _due_date(due_periods[0], due_day)
        tenant_id = add_tenant(
            prop_id, unit, name,
            f"tenant{idx + 1:02d}@example.test",
            f"+91 90000 {idx + 1:05d}", rent, due_day,
            lease_start,
            None,
        )
        tenant_specs.append((tenant_id, rent, due_day, risk_band))

    with db_session() as conn:
        for tenant_id, rent, due_day, risk_band in tenant_specs:
            for period in due_periods:
                due = _due_date(period, due_day)
                # Do not generate transactions for a charge that is still in the future.
                if due > today:
                    continue
                cur = conn.execute(
                    "INSERT OR IGNORE INTO rent_charges(tenant_id,period,due_date,amount_due) VALUES(?,?,?,?)",
                    (tenant_id, period, due.isoformat(), rent),
                )
                charge_id = cur.lastrowid
                if not charge_id:
                    charge_id = conn.execute(
                        "SELECT id FROM rent_charges WHERE tenant_id=? AND period=?", (tenant_id, period)
                    ).fetchone()[0]
                if risk_band == "low":
                    late_probability = 0.07
                elif risk_band == "medium":
                    late_probability = 0.28
                else:
                    late_probability = 0.58
                late = rng.random() < late_probability
                if late:
                    delay = rng.randint(2, 24 if risk_band == "high" else 14)
                    paid_date = due + timedelta(days=delay)
                else:
                    paid_date = due - timedelta(days=rng.randint(0, 3))
                # Leave a small set of recent high-risk charges open to make collections visible.
                is_recent = period == _month_period(today)
                leave_open = is_recent and risk_band == "high" and rng.random() < 0.30
                if paid_date <= today and not leave_open:
                    conn.execute(
                        "INSERT INTO payments(charge_id,paid_on,amount,method,reference) VALUES(?,?,?,?,?)",
                        (charge_id, paid_date.isoformat(), rent, rng.choice(["UPI", "Bank transfer", "NEFT"]), f"DEMO-{tenant_id}-{period}"),
                    )
                elif not leave_open and due < today and rng.random() < 0.08:
                    # A partially paid overdue charge models an instalment.
                    partial = round(rent * rng.uniform(0.25, 0.65), 2)
                    conn.execute(
                        "INSERT INTO payments(charge_id,paid_on,amount,method,reference) VALUES(?,?,?,?,?)",
                        (charge_id, (due + timedelta(days=3)).isoformat(), partial, "UPI", f"DEMO-PART-{tenant_id}-{period}"),
                    )

    ensure_charges_for_months(months=2)
