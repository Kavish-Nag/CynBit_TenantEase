# Assessment Submission Notes

## Primary deliverable

Submit the project ZIP or a public Git repository containing the complete application source and documentation.

## What is included

- Working property-management workflow: properties, tenants, rent charges, partial/full payment ledger and collection dashboard.
- Predictive Intelligence extension: synthetic-data training, model comparison, next-cycle late-risk scoring and explanation.
- Smart reminder drafts for review.
- CSV retraining workflow and export of operational records.
- AI Opportunity Report and Technical Approach Report.
- Architecture diagram, setup instructions and automated tests.

## Suggested execution notes for the assessment form

> TenantEase Predictive Intelligence extends TenantEase's core property and rent-management workflow with an explainable next-cycle late-payment risk model. The prototype includes property/unit tracking, tenant and lease records, monthly rent charges, partial/full payment recording, a payment ledger, collection dashboards, risk-prioritised follow-ups and editable reminder drafts. The ML layer compares Random Forest and Logistic Regression using a tenant-group holdout split on clearly labelled synthetic data. It also supports retraining from a labelled historical CSV. Run `start.bat` on Windows or `start.sh` on macOS/Linux to install dependencies and launch the Streamlit application. The demo is self-contained and is not connected to Cynbit's private production TenantEase APIs or database. Real TenantEase data and approved integration interfaces would be required for a production pilot.

## Before sharing

1. Extract the ZIP and run the app once on the target machine.
2. Confirm the local dashboard opens at `http://localhost:8501`.
3. Keep the phrase **synthetic demo data** in the report and demo narration. Do not present generated records or scores as actual TenantEase customer data.
4. Share the source ZIP/repository using a link accessible to the assessment team. Do not share a local `C:\` file path.
