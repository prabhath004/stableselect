# NeurIPS Workshop Submission Checklist

## Before Running Full Pilot

- [ ] Confirm target workshop after NeurIPS workshop list is announced.
- [ ] Confirm target workshop page limit and anonymity rules.
- [ ] Confirm whether submissions are archival or non-archival.
- [ ] Confirm OpenReview submission deadline.
- [ ] Confirm accepted file format and supplementary material rules.

## Experiment Readiness

- [ ] `configs/models.yaml` has final model IDs.
- [ ] `configs/tasks.yaml` has final task names or datasets.
- [ ] One BF16 sanity run succeeds.
- [ ] One 4-bit sanity run succeeds on CUDA.
- [ ] Raw predictions are saved.
- [ ] Metrics CSVs are saved.
- [ ] Analysis tables are generated.

## Required Result Artifacts

- [ ] Main score table.
- [ ] Rank stability table.
- [ ] Quantization drop table.
- [ ] Deployment Stability Score table.
- [ ] Ranking flip heatmap.
- [ ] Average score vs stability score plot.

## Paper Readiness

- [ ] Four-page NeurIPS-format PDF.
- [ ] Abstract states problem, method, result, and takeaway.
- [ ] Introduction frames model-selection risk.
- [ ] Method is reproducible from configs.
- [ ] Results answer whether rankings flip.
- [ ] Limitations are explicit.
- [ ] README explains how to reproduce the pilot.

## Submission Claim Check

The paper should support one of these claims:

- [ ] The clean baseline winner changes under at least one deployment setting.
- [ ] The highest average model is not the highest DSS model.
- [ ] 4-bit quantization causes uneven degradation across task/language groups.
- [ ] Average score hides a large worst-group drop.

