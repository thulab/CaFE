# Guided Workbench UI Design

## Goal

Optimize the TSBenchmark frontend around a guided evaluation workbench. The first screen should make the CSV-to-report workflow easier to scan, with clear step hierarchy, visible progress, accessible controls, and compact status feedback.

## Scope

This iteration covers the Vue frontend only:

- Evaluation wizard shell and step presentation.
- Wizard step controls, status messages, preview table, model selection, and report link.
- Result pages get the same visual baseline for page shell, controls, tables, loading states, and error messages.

It does not change backend APIs, routing behavior, persistence, or benchmarking logic.

## Design Direction

Use the selected Guided Workbench direction:

- Light analytics SaaS interface with restrained blue data colors and amber action highlights.
- Dense but readable layout for repeated technical workflows.
- Left progress rail on desktop, stacked progress on mobile.
- Step cards with stable dimensions, clear headings, and concise helper copy.
- Native accessible form controls styled consistently; no emoji icons.

## Components

- `EvaluationWizardPage.vue` owns the page shell, header, progress rail, and step cards.
- Wizard components keep behavior local but add semantic headings, labels, helper text, and status classes.
- Result pages use shared shell classes from a global stylesheet instead of duplicate scoped page CSS.
- Result tables are wrapped for mobile overflow and styled for scanning.

## States

- Upload waits for a CSV, then shows file preview and enables `Next`.
- Configure validates one target and positive split values before loading a shard.
- Load shard shows waiting or sample count.
- Track creation is disabled until a shard exists and reports readiness.
- Run selection disables `Run` until at least one model is selected and shows run status.
- Report link appears as a prominent action after a report id is available.
- Result pages show loading states instead of blank screens.

## Testing

Update frontend tests to assert the new accessible shell and existing workflow controls:

- Smoke test renders the workbench title and model-backed run controls.
- Upload test verifies the upload step remains blocked until preview exists.
- Existing wizard and result tests continue to pass.
