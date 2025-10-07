# Cross-Database Generalization Plan

## Context
- Current run configuration (`experiments/localtest.json`) points `ExternalWorkload` to a single mixed workload that is meant for training, while the DSB workload should be consumed only during validation/test.
- During preparation, the observation managers derive feature dimensionality directly from `WorkloadGenerator.number_of_query_classes`, which is currently populated only from the training workload when using the `ExternalWorkload` flag.
- As soon as the DSB workload (with a different query catalog) is introduced at evaluation time, the state vector no longer matches the observation space defined at training start, leading to runtime shape mismatches.

## Goals
1. Allow the config to declare separate external workloads for training and testing while keeping the 
   preparation pipeline aware of both.
2. Generate observation spaces that are stable across training, validation, and testing (union of query classes).
3. Ensure RL training samples **only** the training workload, while evaluation uses the DSB workload.
4. Preserve existing experiment outputs (reports, workload exports, comparison baselines).

## Assumptions
- Both workload JSONs follow the existing labeled workload schema (queries + tables_columns + frequencies).
- It is acceptable to enlarge the observation vector using zero-padding for unseen query classes.
- Environment budgets and embedding sizes can remain unchanged; only the query-class dependent sections need alignment.

## Solution Outline
- Split workload responsibilities at the configuration layer: `ExternalWorkload`/`WorkloadPath` for training, `TestExternalWorkload`/`TestWorkloadPath` for validation & testing.
- During preparation, load **both** workloads, extract the union of query classes/columns, and expose that union to downstream components (observation manager, action manager column catalog, embedders).
- Keep the training dataset driving RL episodes; validation/test iterators should pull from the DSB dataset loaded via the new test path.
- Update serialization/reporting code paths so that artifacts still list evaluation workloads separately.

## Workstreams & Tasks
### 1. Configuration Contract
- [ ] Update `experiments/localtest.json` so that:
  - `ExternalWorkload` = `true`, `WorkloadPath` points to `/home/baiyutao/SWIRL/LocalExp/mixed/tpch_job.2000_1500.1000w.10q.workload.labeled.mixed.tpch_renamed.json`.
  - `TestExternalWorkload` = `true`, `TestWorkloadPath` points to the desired DSB workload file.
  - Retain other knobs (budgets, timesteps, etc.) from the previous working config.
- [ ] Mirror these expectations in README/experiment guide snippets if needed.

### 2. Configuration Parsing & Validation
- [ ] Extend `ConfigurationParser` checks (already partially present) to ensure both training and test paths are validated, and raise clear error messages if either JSON is missing or malformed.
- [ ] Surface final resolved paths in the parsed config (e.g., `config["resolved_workloads"]`) to simplify debugging/logging.

### 3. Workload Generation Pipeline (`swirl/workload_generator.py`)
- [ ] On initialization, load both training and test external files when the respective flags are enabled.
- [ ] Build a **global query catalog** that merges:
  - Training external query definitions.
  - Test external query definitions (DSB).
  - Any internally generated queries (fallback path).
- [ ] Assign stable query numbers across both sets, storing a mapping from original workload descriptors to global IDs so that `query.nr` is consistent.
- [ ] Ensure `self.number_of_query_classes`, `self.query_texts`, and `self.globally_indexable_columns` are derived from the global catalog.
- [ ] Keep `wl_training`, `wl_validation`, and `wl_testing` constructed from their respective sources:
  - Training workloads: only from the training file (or internal generation when external not provided).
  - Validation/Test workloads: from the DSB file when `TestExternalWorkload` is true; otherwise fall back to training remainder.
- [ ] When computing indexable columns (`_store_indexable_columns`, `_select_indexable_columns`), include columns appearing in **either** workload to avoid missing features during testing.

### 4. Observation & Embedding Stack
- [ ] Pass the global query-class count to all observation managers (`SingleColumnIndexObservationManager`, `EmbeddingObservationManager`, etc.) via `observation_manager_config`.
- [ ] When initializing per-workload frequency vectors, rely on the global query ID mapping so DSB queries naturally zero-pad against the training space.
- [ ] For embeddings (`PlanEmbedderLSIBOW` or others), ensure the embedding vocabulary is initialized from the merged query list so that test plans can be embedded without retraining the encoder mid-run.

### 5. Training / Evaluation Loop (`swirl/experiment.py`)
- [ ] Double-check that `make_env` uses:
  - `wl_training` for `EnvironmentType.TRAINING`.
  - `wl_validation[0]` / `wl_testing[0]` for validation/test runs.
  No code change should be necessary, but add assertions/logging to confirm the workload source during runtime.
- [ ] Update progress logging to state which workload file each phase is drawing from, aiding experiment reproducibility.

### 6. Artifact Generation & Reporting
- [ ] Ensure workload export helpers (`_save_workloads_as_json`, `_create_workload_files_with_labels`) preserve the DSB-specific metadata when writing labeled outputs.
- [ ] Include workload source (training vs test path) in the report header for traceability.

### 7. Verification & Tooling
- [ ] Add a lightweight unit/integration test that instantiates `WorkloadGenerator` with both paths and asserts:
  - `number_of_query_classes` equals the union size.
  - Training workloads exclude DSB queries.
  - Validation/test workloads include only DSB entries.
- [ ] Provide a small smoke script (e.g., `scripts/check_workload_shapes.py`) that loads the config and prints observation-space shapes for training vs testing to catch regressions quickly.

## Edge Cases & Risks
- DSB workload introduces query columns unseen during training. Mitigation: global column catalog union + zero-padding frequencies.
- Workload JSON schema deviations (missing `tables_columns`). Mitigation: strict validation with actionable errors.
- Embedding cache size growth due to union. Mitigation: monitor memory usage; permit configuration to limit test-only embeddings if needed.

## Validation Checklist
1. `python main.py --config experiments/localtest.json` completes without observation shape errors.
2. Training logs show workloads from the mixed dataset; evaluation logs show DSB workloads.
3. Exported workload JSONs contain entries from both datasets with correct labels.
4. Added unit test(s) pass under the project’s test runner.

## Deliverables
- Updated configuration (`experiments/localtest.json`).
- Code changes across `ConfigurationParser`, `WorkloadGenerator`, observation manager setup, and related utilities.
- Optional helper script + documentation updates.

## Follow-ups
- Consider introducing a config section that explicitly names datasets (e.g., `"datasets": {"train": ..., "test": ...}`) to scale to more splits.
- Evaluate whether validation should also point to DSB or remain mixed; adjust budgets accordingly.
- Profile embedding initialization time when both workloads are large; cache results if needed.
