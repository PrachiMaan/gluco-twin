# Model card: GlucoTwin event model

## What the model does

GlucoTwin predicts, at a given 5-minute checkpoint, whether the patient will
spend at least 30 consecutive minutes above 180 mg/dL at some point in the next
2 hours. It also forecasts glucose 30, 60 and 120 minutes ahead. The output is a
probability plus three numbers, not a diagnosis.

The model is a research prototype. It is **not** a medical device, has not been
clinically validated, and must never be the only reason a treatment decision is
made.

## Problem setup

| Item | Value |
| --- | --- |
| Unit of prediction | one 5-minute checkpoint per patient |
| Positive label | at least 30 consecutive minutes > 180 mg/dL inside the next 120 minutes |
| Negative label | otherwise |
| Base rate in the test cohort | 41.1% of checkpoints positive |
| Forecast targets | glucose at +30, +60 and +120 minutes |
| Split | by patient, 900 train / 300 validation / 300 test, seed 42 |
| Test windows evaluated | 200,400 (15-minute checkpoints) |

Splitting by patient is the important control. Windows from the same patient are
highly correlated, so a random window split would leak patient identity and
inflate every metric.

## Models compared

| Name | Input | Role |
| --- | --- | --- |
| `persistence` | current glucose plus threshold | baseline; fires when glucose is already above the threshold |
| `lgbm_cgm` | 15 CGM features (the `g_` block, without the fraction-above-range block) | CGM-only LightGBM |
| `lgbm_ehr` | 20 static EHR fields | EHR-only LightGBM |
| `lgbm_fused` | all 55 features | deployed model |
| `fusion_net` | 6 hours of 5-channel time series + 20 static EHR fields | two-branch neural model |

### Features (55)

- 19 CGM features: current value, means over 30 minutes and 1, 3 and 6 hours,
  standard deviation, coefficient of variation, min, max, range, three slopes,
  two deltas, fraction above 180, fraction above 250, fraction below 70, missing
  fraction.
- 13 wearable features: heart rate now, 1-hour mean and max, 1-hour delta, HRV
  now, 1-hour mean, 1-hour to 6-hour HRV ratio, night HRV, steps over 1 and 3
  hours, asleep fraction over 1 and 6 hours, deep-sleep fraction over 6 hours.
- 3 context features: sine and cosine of the hour, night flag.
- 20 static EHR fields: age, sex, BMI, waist, vegetarian diet, family history,
  years since diagnosis, HbA1c, fasting glucose, eGFR, LDL, triglycerides,
  systolic BP, four medication flags, hypertension, TCF7L2 dose, prior
  hypoglycemia count.

CGM gaps are linearly interpolated per patient and a missing fraction is kept as
a feature, so the model knows when it is looking at imputed values.

### LightGBM settings (deployed model)

250 trees, learning rate 0.06, 31 leaves, minimum 200 rows per leaf, feature
fraction 0.85, bagging fraction 0.8, L2 penalty 1.0, one thread, deterministic
mode, seed 42. Class imbalance is handled with `scale_pos_weight` equal to the
negative to positive ratio (1.48 on the training windows).

### Fusion network

A two-branch network implemented in numpy (`src/models.py`):

- Sequence branch: 5 channels (glucose, heart rate, HRV, steps, sleep stage) at
  10-minute resolution over 6 hours (36 steps), 5 dilated causal convolution
  layers with 16 channels per layer, GELU activations, layer normalisation and
  global average plus last-step pooling.
- Static branch: 20 EHR fields into a 32-unit MLP.
- Joint head: both branches concatenated, 32 hidden units, then a shared
  representation with one sigmoid event output and three linear forecast
  outputs.
- Training: Adam, learning rate 2e-3 with a 0.5 decay per epoch, batch 128,
  2 epochs, a fixed 110,000 sampled windows per epoch. Manual backpropagation,
  gradient-checked against finite differences in `tests/test_models.py`.

### Probability calibration

Isotonic regression is fitted on the validation split and applied to the test
predictions. Brier scores and the calibration plot in
`results/plots/calibration.png` use the calibrated probabilities.

### Alarm protocol

An alarm is the first checkpoint of a run above the operating threshold. A
2-hour cooldown suppresses repeat alarms for the same event. An alarm counts as
a true alarm if it lands within the 2 hours before the start of a real
hyperglycemic run. Alarms raised while the patient is already above 180 mg/dL are
excluded from the false-alarm count, because the event has already happened and
the alert is no longer a prediction. The threshold is chosen on validation to
keep false alarms at or below 0.5 per patient-day, then applied unchanged to the
test set.

## Results (test patients only, seed 42)

| Model | AUROC | AUPRC | Within-patient AUROC | Brier | Sensitivity at 0.5 FA/day | Mean lead (min) |
| --- | --- | --- | --- | --- | --- | --- |
| Persistence | 0.7972 | 0.7984 | 0.6213 | 0.1889 | 6.4% | 47.6 |
| LightGBM, CGM only | 0.9837 | 0.9797 | 0.9683 | 0.0460 | 70.4% | 50.2 |
| LightGBM, EHR only | 0.7906 | 0.7635 | 0.5000 | 0.1732 | 4.5% | 50.2 |
| LightGBM, fused (deployed) | 0.9941 | 0.9921 | 0.9882 | 0.0283 | 77.9% | 53.5 |
| Fusion network | 0.9496 | 0.9333 | 0.9159 | 0.0885 | 24.2% | 47.2 |

The ablation is computed at the same 30-minute checkpoint cadence for every
model. The deployed model was also evaluated on the denser 15-minute timeline:
AUROC 0.9940, AUPRC 0.9920, Brier 0.0284, sensitivity 81.2% at 0.62 false
alarms per patient-day, mean lead 59.8 minutes.

Glucose forecasts (LightGBM regressors, 150 trees):

| Horizon | RMSE (mg/dL) | MAE (mg/dL) |
| --- | --- | --- |
| +30 min | 11.0 | 7.5 |
| +60 min | 13.9 | 9.1 |
| +120 min | 23.4 | 15.2 |

The fusion network's forecast RMSE is 31 to 32 mg/dL at all three horizons and
its event AUROC is 0.9496, below LightGBM. It is **not** the deployed model. The
honest reading is that a small numpy TCN with two epochs of training, trained on
a CPU budget, did not beat a well-tuned gradient-boosted tree on this synthetic
dataset. The tree model is shipped, and the network stays in the repository as
the comparison the brief asked for.

### What drives the predictions

SHAP values from the deployed tree model, mean absolute contribution on the test
set:

| Feature | Mean abs SHAP |
| --- | --- |
| Fraction above 180 in the last 3 hours | 2.24 |
| Maximum glucose in the last 3 hours | 1.28 |
| Mean glucose in the last 6 hours | 0.93 |
| Hour of day (cosine) | 0.75 |
| Mean glucose in the last 1 hour | 0.61 |
| Glucose delta over 60 minutes | 0.54 |
| Glucose slope over 60 minutes | 0.42 |

The ordering is clinically plausible: recent time above range and the current
excursion dominate, with time of day second. Static EHR fields have small SHAP
values because they explain *who* is at risk, not *when* the next event happens.
That is also why the EHR-only model has a within-patient AUROC of exactly 0.5:
a static vector cannot rank two moments in the same patient's day. Its overall
AUROC of 0.79 mostly reflects differences between patients.

## Intended use

- Demonstration of a clinical decision-support concept at a hackathon.
- Teaching and research on multimodal diabetes data.
- Testing what-if scenarios in a simulated cohort.

## Out of scope

- Diagnosis, treatment adjustment, insulin dosing, or any autonomous action.
- Real patients, real CGM devices, or real clinical workflows.
- Pediatric, type 1, or gestational diabetes.
- Patients with eGFR below 30, recent acute illness, or steroid-induced
  hyperglycemia.
- Deployment without prospective validation and regulatory review.

## Limitations and risks

1. **Synthetic data.** Every metric is measured on data produced by the same
   simulator family used to create the cohort. The model may be learning
   simulator regularities that real CGM data does not share. No claim of clinical
   accuracy is made.
2. **High event rate.** 41% of checkpoints are positive, far above a real clinic
   panel, because the cohort is deliberately uncontrolled. Sensitivity and AUPRC
   will drop on a well-controlled population.
3. **No external validation.** `src/validate_external.py` is provided for an
   optional open dataset (CGMacros or Shanghai T2DM), but it has not been run
   here because the data is not bundled. Until it is run, external performance
   is unknown.
4. **Subgroup fairness is not established.** Sex, age band, diet and genotype
   subgroups were not evaluated for equal sensitivity or calibration. This is
   required work before any clinical use.
5. **Lead time vs false alarms.** At the operating point the model raises about
   2.7 alarms per patient-day at the dense cadence. That is acceptable for a
   demo but too noisy for a ward. A real deployment needs a higher threshold or
   clinician-tuned alert budgets.
6. **Static EHR staleness.** Labs are treated as current for 14 days. Real EHR
   data would be older and sometimes missing.
7. **No CGM device model.** Sensor lag, calibration drift and compression lows
   are approximated by Gaussian noise and dropouts.
8. **Calibration drift.** Isotonic calibration is fitted on 300 validation
   patients from the same generator. It will not transfer unchanged to a new
   site.

## Ethical notes

- Human in the loop: the dashboard presents evidence and a suggested scenario.
  It never changes therapy.
- The what-if engine is a simulation using the patient's fitted parameters. Its
  outputs are labelled as simulated, not prescribed.
- A false negative is a missed chance to intervene, not proof that nothing will
  happen. The UI states this.
- Under India's DPDP Act, 2023 and HIPAA, real deployment would require consent,
  purpose limitation, audit logs and a data-fiduciary agreement. This repository
  uses synthetic data only, so it has no personal data to protect.

## Reproduction

```
python src/train.py --trees
python src/train.py --forecast
python src/train.py --net
python src/evaluate.py
```

All figures in this card come from `results/metrics.json`,
`results/ablation.csv`, `results/shap_importance.csv` and
`results/param_recovery.json`, written by those commands. The model files live
in `models/` and are not committed.
