# AI Opportunity Report
## TenantEase Predictive Intelligence: Predictive Rent Collection and Risk Intelligence

**Prepared for:** Cynbit Technologies, AI/ML Engineer Intern assessment  
**Product assessed:** TenantEase  
**Prototype:** TenantEase Predictive Intelligence  
**Evidence boundary:** Publicly available Cynbit product information; no access to private TenantEase data or implementation was provided.

---

## 1. Executive summary

TenantEase is publicly described as a smart property and tenant management platform. Its advertised capabilities include property and unit tracking, tenant profiles, rent and payment tracking, pending dues, transaction history, system-driven reminders and business dashboards for occupancy and revenue trends.

This creates a natural opportunity to move from **reactive rent administration** to **proactive rent-collection planning**. TenantEase Predictive Intelligence adds an explainable late-payment risk estimate for the next rent cycle, surfaces the historical signals behind each estimate, prioritises follow-up work and generates a human-reviewable reminder draft.

The proposal is deliberately an extension of the existing workflow. It does not replace tenant management, rent billing, payment recording or reminders. Those operational records become the source of features for the predictive layer.

**Proposed value:** help property managers decide who may need an earlier, more personal follow-up, while keeping all decisions with a human operator.

## 2. Product and business context

Cynbit publicly positions TenantEase as its smart property and tenant management platform. The product page describes these capabilities:

- **Financial control:** monthly rent and payment intelligence, transaction history and pending dues.
- **Automated reminders:** system-driven rent follow-up.
- **Unit tracking:** building/room inventory and availability.
- **Tenant profiles:** structured tenant records and historical information.
- **Business dashboard:** occupancy and revenue trends.

These functions already produce operational data that can support analytics and predictive modelling. The opportunity is therefore adjacent to the core product, rather than an unrelated AI feature.

### Existing workflow and proposed extension

| Existing TenantEase workflow | Proposed intelligence layer |
|---|---|
| Store tenant, property and lease information | Use lease age and rent amount as model inputs |
| Track monthly rent charges | Predict the likelihood of late settlement for the next cycle |
| Record payment history | Derive rolling payment-behaviour features |
| Show pending and overdue dues | Prioritise follow-ups using risk and outstanding exposure together |
| Send or automate reminders | Create context-aware reminder drafts for human review |
| Show occupancy and revenue dashboards | Add risk distribution and risk-weighted upcoming rent exposure |

The public product description does not establish whether Cynbit has unadvertised predictive capabilities. This report identifies an opportunity based on the capabilities currently presented publicly, not a claim that such features do not exist internally.

## 3. Problem statement

Property managers can see whether a rent payment is paid, pending or overdue. However, a purely status-based workflow may only surface a problem after a due date has passed. It can also make it difficult to prioritise a large portfolio when many payments are approaching at once.

The proposed system asks:

> Using a tenant's recorded payment history and the upcoming rent-cycle details, can TenantEase estimate the likelihood of a late payment early enough for a property manager to plan an appropriate follow-up?

This is a supervised binary-classification problem. The target is whether a rent charge is settled after the configured grace period, or remains unsettled beyond that period.

## 4. Proposed solution

### 4.1 Predictive scoring

For each active tenant with an upcoming rent charge, the model estimates:

`P(late_next_cycle = 1 | prior payment history, rent and cycle details)`

The interface displays a probability and a configurable demonstration band:

- **Low:** below 35%
- **Medium:** 35% to below 65%
- **High:** 65% or above

These cut-offs are prototype defaults, not validated business policy. They should be calibrated against real tenant payment outcomes and the cost of false positives/false negatives.

### 4.2 Explainability and action support

The application exposes the main input signals and readable reasons such as previous late-payment rate, consecutive late cycles, average delay and unsettled past cycles. It suggests operational actions such as reconciling a balance or sending an earlier reminder.

The prediction is not a statement about a tenant's intent or ability to pay. It is a limited estimate based on recorded payment events. Low history coverage is explicitly highlighted.

### 4.3 Reminder drafting

A manager can choose an outstanding charge and generate a Friendly, Professional or Firm reminder draft. The draft includes the tenant, property/unit, billing period, due date and outstanding amount. The message is editable and downloadable. The system does not send it automatically and does not include the internal risk score in tenant-facing copy.

## 5. Data requirements

### Minimum operational data

- Tenant and property identifiers
- Lease start/end dates and active status
- Monthly rent and due day
- Rent-charge period, due date and amount due
- Payment transaction date and amount
- Charge settlement status

### Derived features

| Feature | Definition | Why it may help |
|---|---|---|
| Previous late rate | Late cycles divided by observed cycles in the last six cycles | Captures recent payment consistency |
| Average delay days | Mean delay across late cycles in the last six cycles | Captures typical delay magnitude |
| Late streak | Consecutive late cycles immediately before the target cycle | Captures persistent recent patterns |
| Previous missed count | Unsettled overdue cycles in the last six observed cycles | Captures unresolved balances |
| Recent delay days | Delay in the most recent observed cycle | Captures the latest behaviour |
| Payments observed | Number of completed/observable prior cycles | Supports history-coverage messaging |
| Rent amount | Contracted monthly rent | Allows evaluation across different rent levels |
| Due day | Calendar day of the next cycle's due date | Captures cycle timing |
| Lease months | Approximate time since lease start | Separates new/limited-history tenancies from established records |

Only cycles before the target due date are used to create prediction features. This prevents the target cycle's result from leaking into its own prediction.

## 6. Expected business impact and measurement

Potential operational outcomes include:

1. Earlier visibility into upcoming late-payment exposure.
2. More consistent prioritisation across properties and units.
3. More relevant follow-up timing and message context.
4. Better visibility into whether historical risk signals translate into actual late outcomes.

These are hypotheses to validate, not guaranteed improvements. A pilot should compare the new workflow against the existing reminder process.

### Suggested pilot KPIs

| KPI | Definition |
|---|---|
| Late-payment recall | Share of late cycles identified as risk-positive before the due date |
| PR-AUC | Ranking quality for the late-payment class, especially when late payments are less common |
| Brier score | Calibration quality of predicted probabilities |
| Early-follow-up coverage | Share of high-risk upcoming charges reviewed before their due dates |
| On-time collection rate | Share of charges fully settled by due date or agreed grace date |
| Manager review time | Time required to prioritise the upcoming collection queue |
| False-positive review rate | Share of high-risk alerts followed by on-time settlement |

Track performance by property and over time. Monitor for changes in collection policy, rent cycles, seasonality and data completeness.

## 7. Alternatives considered

| Opportunity | Fit with public product workflow | Assessment |
|---|---|---|
| TenantEase predictive rent-risk scoring | Directly extends payment history, rent tracking and reminders | Selected: clear structured-data ML problem and measurable outcome |
| TenantEase maintenance-ticket classification | Potentially useful but maintenance workflows are not prominent in the public product description | Less directly supported by advertised scope |
| Revio review-topic intelligence | Revio already advertises sentiment analysis, response generation, crisis detection and multi-location reputation management | Harder to distinguish without private product knowledge |
| Cynbit Studio job-skill recommendation | Studio already advertises adaptive learning, AI evaluation, AI mentors and job-matching tools | Valuable but overlaps with a broad set of advertised AI features |

TenantEase was selected because the proposed feature naturally uses data that the existing product is publicly described as collecting, and its success can be measured with payment outcomes.

## 8. Feasibility and prototype scope

The delivered proof of concept includes:

- Property portfolio and unit-capacity management.
- Tenant profile and lease information.
- Monthly rent charges and partial/full payment recording.
- Payment ledger, collection dashboard and overdue calculation.
- Synthetic historical payment data for a reproducible demonstration.
- Random Forest and Logistic Regression model comparison.
- Next-cycle risk scoring, explanations and risk prioritisation.
- Contextual reminder drafting.
- CSV-based custom model training and operational data export.

The demo model is trained on synthetic records. No real TenantEase customer data was available. The model workflow and integration are demonstrated, but production performance cannot be claimed until representative historical data is used.

## 9. Responsible-use and security considerations

- Keep risk scores as decision-support signals only.
- Do not automatically penalise tenants, deny services or take legal/financial action based on a score.
- Show the reason and history coverage for each estimate.
- Give property managers a path to correct inaccurate payment records.
- Avoid sensitive demographic or protected attributes as predictive features.
- Obtain appropriate permission and purpose limitation for historical payment data.
- Use secure authentication, role-based access, encryption, audit trails, retention controls and backups in production.
- Validate that error rates do not create unfair treatment across properties or tenant groups.

## 10. Conclusion

TenantEase Predictive Intelligence is a practical next step for a platform already centred on property records, rent collection and reminders. It turns payment history into an early-warning workflow while preserving the existing operational system and keeping a property manager in control.

The next step for Cynbit would be a controlled pilot using de-identified, consented historical TenantEase records, with a time-aware evaluation and a review of operational outcomes before a production rollout.

## Public source

Cynbit Technologies, **Proprietary Software Suite: Cynbit Studio, TenantEase & Revio**, product page: https://www.cynbit.in/products (accessed 30 October 2026).
