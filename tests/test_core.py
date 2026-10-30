import os
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

TEMP_DIR = tempfile.TemporaryDirectory()
os.environ["TENANTEASE_DB_PATH"] = str(Path(TEMP_DIR.name) / "test_tenantease.db")

from tenantease import db  # noqa: E402
from tenantease.features import build_tenant_features  # noqa: E402
from tenantease.ml import make_synthetic_training_data, train_and_evaluate  # noqa: E402


class TenantEaseCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def setUp(self):
        db.clear_all_data()

    def test_property_tenant_and_charge_creation(self):
        property_id = db.add_property("Test House", "Pune", 4)
        tenant_id = db.add_tenant(
            property_id, "A-101", "Demo Tenant", "demo@example.test", "", 15000, 5,
            date.today() - timedelta(days=50), None,
        )
        first = db.ensure_charges_for_months(2)
        second = db.ensure_charges_for_months(2)
        self.assertGreaterEqual(first, 1)
        self.assertEqual(second, 0)
        self.assertIn(tenant_id, [t["id"] for t in db.get_tenants(active_only=True)])

    def test_partial_then_full_payment_updates_charge(self):
        property_id = db.add_property("Payment Test", "Pune", 2)
        tenant_id = db.add_tenant(
            property_id, "B-201", "Payment Tenant", "", "", 12000, 5,
            date.today() - timedelta(days=90), None,
        )
        db.ensure_charges_for_months(2)
        charge = next(c for c in db.get_charge_rows() if c["tenant_id"] == tenant_id and c["outstanding"] > 0)
        db.record_payment(charge["charge_id"], date.today(), 5000, "UPI", "TEST-PART")
        partial = next(c for c in db.get_charge_rows() if c["charge_id"] == charge["charge_id"])
        self.assertEqual(partial["status"], "Partial")
        self.assertAlmostEqual(partial["outstanding"], 7000)
        db.record_payment(charge["charge_id"], date.today(), 7000, "Bank transfer", "TEST-FINAL")
        paid = next(c for c in db.get_charge_rows() if c["charge_id"] == charge["charge_id"])
        self.assertEqual(paid["status"], "Paid")
        self.assertAlmostEqual(paid["outstanding"], 0)

    def test_feature_builder_uses_prior_cycles_only(self):
        property_id = db.add_property("Feature Test", "Pune", 2)
        start = date.today() - timedelta(days=100)
        tenant_id = db.add_tenant(property_id, "C-301", "Feature Tenant", "", "", 18000, 5, start, None)
        tenant = db.get_tenant(tenant_id)
        due = date.today() - timedelta(days=35)
        with db.db_session() as conn:
            cur = conn.execute(
                "INSERT INTO rent_charges(tenant_id,period,due_date,amount_due) VALUES(?,?,?,?)",
                (tenant_id, "2026-08", due.isoformat(), 18000),
            )
            conn.execute(
                "INSERT INTO payments(charge_id,paid_on,amount,method,reference) VALUES(?,?,?,?,?)",
                (cur.lastrowid, (due + timedelta(days=8)).isoformat(), 18000, "UPI", "FEATURE-LATE"),
            )
        target_due = date.today() + timedelta(days=35)
        features = build_tenant_features(tenant, target_due, as_of_date=date.today())
        self.assertAlmostEqual(features["previous_late_rate"], 1.0)
        self.assertAlmostEqual(features["avg_delay_days"], 8.0)
        self.assertEqual(features["late_streak"], 1.0)

    def test_synthetic_training_and_model_evaluation(self):
        df = make_synthetic_training_data(n_tenants=100, cycles=8, seed=123)
        self.assertGreaterEqual(len(df), 80)
        self.assertEqual(set(df["late_next_cycle"].unique()), {0, 1})
        bundle = train_and_evaluate(df, group_col="tenant_group")
        self.assertIn(bundle["model_name"], {"Random Forest", "Logistic Regression"})
        self.assertIn("pr_auc", bundle["metrics"][bundle["model_name"]])
        self.assertEqual(len(bundle["feature_importances"]), 9)


if __name__ == "__main__":
    unittest.main()
