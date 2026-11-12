# Monitoring notes — Telco churn prediction

## What is logged

Every successful `POST /predict` appends one JSON object (one JSON Lines record) to `logs/predictions.log`. Each event contains:
- UTC timestamp
- a compact feature summary: `tenure`, `Contract`, `MonthlyCharges`, `InternetService`, and `PaymentMethod`
- churn probability
- predicted `Yes`/`No` label

The log deliberately omits direct customer identifiers. In production, access to logs should be restricted and retention defined. The Docker Compose setup mounts `./logs` into the backend container so logs survive container recreation on the same host. Ephemeral free hosting filesystems may reset; use a persistent disk or managed log sink for durable production monitoring.

## What drift could look like

**Data drift** means the distribution of incoming customer features changes relative to the data used to train the model. For this model, examples include:
- a growing share of `Month-to-month` contracts versus one- or two-year contracts
- the `MonthlyCharges` distribution shifting upward or downward
- changes in the proportions of `Fiber optic`, `DSL`, and `No` internet service
- changes in `PaymentMethod` mix

**Prediction/concept drift** is different. Prediction drift is a change in the distribution of model scores or predicted churn labels. Concept drift occurs when the relationship between customer features and actual churn changes. Since this API does not receive the eventual true churn outcome, concept drift cannot be confirmed from prediction logs alone; outcomes must later be joined from an approved source.

## First checkable warning signals

A scheduled analysis can compare weekly log windows with a training/reference baseline:
1. Population Stability Index (PSI) or quantile comparisons for `MonthlyCharges` and `tenure`.
2. Proportion differences for `Contract`, `InternetService`, and `PaymentMethod`.
3. Mean/median churn probability and the share of `Yes` predictions.

For example, if the weekly share of month-to-month contracts rises sharply and average churn probability also increases, this is an early investigation signal, not proof that model accuracy has fallen. Once labels mature, monitor precision, recall, calibration, and PR-AUC on recent labeled cohorts.

## Demo vs monitored service

This project is a demonstrable inference service with structured request logging. It is **not** a complete monitoring platform: it does not yet calculate drift metrics automatically, alert, or retrain. Public demo deployment should use synthetic/example customer data only. Do not put real customer personal data into a public demo without authorization, security review, and an appropriate privacy basis.
