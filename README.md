# TenantEase Predictive Intelligence

**An integrated property-management and rent-collection prototype with explainable late-payment risk estimates.**

This prototype extends the main TenantEase workflow rather than presenting a disconnected ML screen. Property records, tenant profiles, rent charges, payment transactions, outstanding balances, collection summaries, reminder drafts and predictive risk estimates share one local SQLite database.

> **Prototype boundary:** This is a self-contained extension demo built from the publicly described TenantEase workflow. It is not connected to Cynbit's private/production TenantEase codebase, database or APIs. No internal API or schema was supplied for this assessment. A production integration would use Cynbit-approved interfaces and authentication.

## What works

### Core property-management workflow
- Create and manage properties with unit capacity and locality.
- Add tenant profiles, assign a unit, set monthly rent, rent due day and lease dates.
- Generate current and next-cycle rent charges for active leases.
- Record full or partial rent payments against a charge.
- View charge status as Paid, Partial, Pending or Overdue.
- Review payment history and export properties, tenants, charges and payments.
- Track occupancy, billed rent, collection totals and overdue balances.

### Predictive Intelligence
- Train and evaluate Random Forest and Logistic Regression candidates on a clearly labelled synthetic payment-history dataset on first launch.
- Predict the probability that an upcoming rent cycle will be late.
- Display Low, Medium and High risk bands, history coverage, input features and operational reasons/actions.
- Show holdout metrics including PR-AUC, ROC-AUC, precision, recall, accuracy and Brier score.
- Upload a labelled historical CSV to train and activate a custom model. Evaluation uses a tenant-group split when a `tenant_group` column is available.
- New payments recorded in the rent ledger are used in the feature history for the next prediction run.

## Run manually

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Suggested demo walkthrough

1. **Overview:** show occupancy, collection performance, overdue exposure and the next-cycle risk queue.
2. **Properties & Tenants:** open the portfolio and tenant directory. Add a test property or tenant if useful.
3. **Rent & Payments:** open the charge ledger, record a partial payment, then record the remaining amount. The charge status changes to Paid once the charge is settled.
4. **Predictive Intelligence:** show the next-cycle risk distribution, individual prediction explanations and history coverage.
5. **Smart Follow-ups:** create a reminder draft for an overdue or upcoming charge. Edit and download it, but do not send it from the prototype.
6. **Data & Model:** show model comparison metrics, the training CSV schema and the operational-data export.

## Predictive model data contract

A labelled training CSV must include these numeric columns:

| Column | Meaning |
|---|---|
| `rent_amount` | Monthly rent amount |
| `previous_late_rate` | Fraction of the previous six observed cycles that were late |
| `avg_delay_days` | Average delay on late cycles in the previous six observed cycles |
| `late_streak` | Consecutive late cycles immediately before the target cycle |
| `payments_observed` | Number of completed/observable prior cycles |
| `previous_missed_count` | Number of unsettled, overdue cycles in the previous six observed cycles |
| `recent_delay_days` | Delay in the most recent observed cycle, or zero if on time |
| `due_day` | Calendar day rent is due |
| `lease_months` | Approximate lease age at the target cycle |
| `late_next_cycle` | Target: `1` if next cycle was late, `0` otherwise |
| `tenant_group` | Optional tenant identifier used to keep each tenant in only one evaluation split |

Use the in-app **Download sample training CSV** button as a schema example. Its values are synthetic and must not be represented as real customer data.

## Project structure

```text
TenantEase_Predictive_Intelligence/
├── app.py                         # Streamlit application and UI
├── tenantease/
│   ├── db.py                      # SQLite schema, CRUD, charges, payment ledger, seed data
│   ├── features.py                # Leakage-conscious tenant history features + explanations
│   ├── ml.py                      # Synthetic data, model comparison, training, scoring
│   └── reminders.py               # Local reminder draft generation
├── data/                          # Local database/model files are created here at runtime
├── docs/
│   ├── AI_OPPORTUNITY_REPORT.md
│   ├── TECHNICAL_APPROACH.md
│   └── ARCHITECTURE.md
├── tests/
│   └── test_core.py
├── requirements.txt
├── start.bat
└── start.sh
```

## Data and security notes

- The first run creates a local SQLite database at `data/tenantease.db` and seeds synthetic demo records. The demo names, addresses, payment history, transaction references and contact details are fictitious.
- The custom model is saved locally at `data/custom_model.joblib` after a CSV upload.
- This assessment prototype has no login, role-based access, cloud backup, encryption-at-rest configuration or payment-gateway integration. Do not enter real tenant identity documents or sensitive customer records.
- For production, add authentication and role-based access, tenant-level data isolation, encryption, audit logging, consent/retention controls, secure secrets management, backups and monitoring. Use a verified database/API integration approved by Cynbit.
- Prediction scores are not guarantees and must not be used for tenant punishment, denial of housing or automatic financial decisions. Provide human review and a correction process.
- The synthetic demo model proves the workflow and code path. It does **not** establish production predictive performance. Retrain, calibrate and validate on representative, consented historical TenantEase data before deployment.

## Tests

With dependencies installed, run:

```bash
python -m unittest discover -s tests -v
```

The tests use a temporary SQLite file and do not modify the demo database.

## Technology

Python, Streamlit, SQLite, pandas, NumPy, scikit-learn, Plotly and joblib.
