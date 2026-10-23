# GlucoTwin demo video script

Target length: **22 minutes**. The submission minimum is 20, so the last two
minutes are a buffer. If you run long, trim the data slides, never the live
demo.

Record in one take if you can. If you need cuts, cut between slides, not in the
middle of a click.

Before recording, read the two-minute checklist at the end.

**Presenter:** [MEMBER 1]
**Screen:** 1920 x 1080, browser at 100%, bookmarks bar hidden, notifications
off. The dashboard is designed for a wide screen.

---

## 0:00 to 0:45. Slide 1. Open cold.

**On screen:** title slide.

**Say:**

> Hi, we are [TEAM NAME] from [COLLEGE]. We built GlucoTwin, a digital twin for
> type 2 diabetes. It watches a patient's last six hours and warns the clinic
> two hours before a hyperglycemic event. Then it lets the clinician test what
> would actually help that patient, before saying it.

Do not read the slide. Do not list the tech stack yet.

## 0:45 to 2:30. Slides 2 and 3. The problem.

**On screen:** slide 2, then slide 3.

**Say on slide 2:**

> India has 101 million people with diabetes and another 136 million with
> prediabetes. [CITE] Most of them see a doctor two or three times a year. The
> disease acts every day in between.

**Click:** next slide.

**Say on slide 3:**

> The dangerous hours are the ones nobody sees. Glucose above 180 for half an
> hour does damage quietly, and the worst window is usually after the evening
> meal. A fasting reading at nine in the morning cannot show it. The next HbA1c
> arrives three months later.

Pause. Let that land.

## 2:30 to 4:00. Slide 4. The twin.

**On screen:** slide 4.

**Say:**

> So we rebuilt the patient as a twin. One side is the static record: labs,
> medications, genotype, years since diagnosis. The other is fourteen days of
> CGM, heart rate, HRV, steps and sleep. The twin produces a calibrated
> two-hour risk, a glucose forecast, and a what-if sandbox that runs on this
> patient's own fitted physiology.

> The clinician sees one screen: why the alert fired, and what would change it.

## 4:00 to 5:30. Slides 5 and 6. Data.

**On screen:** slide 5.

**Say:**

> The cohort is 1,500 patients, 14 days each, one reading every five minutes.
> That is 6.05 million glucose readings. Every model in this video trains on one
> CPU core in this room.

**Click:** next slide.

**Say on slide 6:**

> This is synthetic, and that is a design choice, not a shortcut. We have no
> real patient data, so there is no consent problem and nothing to leak.

> The important part is that the record drives the physiology. Fasting glucose
> sets the baseline. BMI, duration of disease and the TCF7L2 genotype set
> insulin sensitivity and meal handling. So the two streams are actually linked,
> not two random datasets joined by a patient ID.

> The cost is real: nothing here proves performance on real patients. We will
> come back to that.

## 5:30 to 6:45. Slide 7. Architecture.

**On screen:** architecture diagram.

**Say:**

> One pipeline. Two synthetic sources, a five-minute alignment step, 55
> features, a fused model, the twin, the what-if engine, the API and the clinic
> screen.

> The optional external validation sits off to the side. Nothing in the main
> pipeline depends on it.

Point at the deployed model box when you say "fused model". Do not read every
box.

## 6:45 to 8:15. Slides 8 and 9. Label and features.

**On screen:** slide 8.

**Say:**

> The label is a real clinical event, not a synthetic score. One if glucose
> stays above 180 for at least thirty minutes inside the next two hours.

> 41% of test windows are positive, because the cohort is deliberately
> uncontrolled. A real primary-care panel would be lower and harder. We split by
> patient, 900, 300, 300, so no patient appears in two splits.

**Click:** next slide.

**Say on slide 9:**

> Fifty-five features in four blocks. Nineteen from CGM: level, means,
> variability, slopes, time above range, and the share of missing readings.
> Thirteen from wearables. Three clock features. Twenty from the record.

> The neural branch also gets a six-hour tensor with five channels.

## 8:15 to 9:45. Slides 10 and 11. Models and the honest loss.

**On screen:** slide 10, the model table.

**Say:**

> We compared five models: persistence, LightGBM on CGM only, LightGBM on the
> record only, LightGBM fused, and the two-branch fusion network from the brief.

> The fused tree wins: AUROC 0.994, 78% sensitivity at about half a false alarm
> per patient-day.

**Click:** slide 11.

**Say:**

> The network was supposed to win. It did not. AUROC 0.9496 against 0.9941. It
> is a real two-branch model, a dilated TCN plus an MLP, written in numpy and
> gradient-checked, but two epochs on one CPU core is not enough to beat 250
> boosted trees.

> So we deployed the tree and kept the network in the ablation. The brief asked
> us to say so if this happened, and this is us saying so.

## 9:45 to 11:15. Slide 12. Results.

**On screen:** slide 12, four big numbers.

**Say:**

> On 300 patients the model never saw: AUROC 0.994, AUPRC 0.992, Brier 0.028,
> 78% sensitivity at the operating point, with a mean warning of 53 minutes.

> On the denser 15-minute timeline it reaches 81% sensitivity with a 60-minute
> mean lead. That is the number to remember: one hour of warning, not five
> minutes.

> The false alarm rate is 0.45 per patient-day in development. The deployed
> dense threshold produces about 2.7. That is a real limitation, not a footnote.

## 11:15 to 12:45. Slide 13. Ablation.

**On screen:** ablation chart.

**Say:**

> Every input helps, and the sensor helps most. Persistence is 0.797. CGM alone
> gets to 0.984. The record alone is 0.791.

> The record-only model has a within-patient AUROC of exactly 0.5. It can tell
> you who is at risk, but not when. That is the whole argument for the dynamic
> stream.

> The fusion net sits at 0.950. The fused tree is the best of both.

## 12:45 to 14:00. Slide 14. Calibration and forecast.

**On screen:** calibration plot.

**Say:**

> A ranking is not enough. The probability has to mean something, so we fit
> isotonic calibration on the validation patients. Brier is 0.028.

> The forecast errors are 11, 14 and 23 mg/dL at 30, 60 and 120 minutes. The
> two-hour number is honest: it is hard, and 23 mg/dL is not a promise.

## 14:00 to 15:00. Slide 15. SHAP.

**On screen:** SHAP summary.

**Say:**

> Recent time above range and the size of the current excursion dominate.
> Time of day is next. The record fields are small, and that is the point: the
> record explains who is at risk, the stream explains when.

> The dashboard turns these into five plain sentences, not feature names.

## 15:00 to 16:15. Slide 16. The twin.

**On screen:** slide 16.

**Say:**

> The twin fits insulin sensitivity and basal glucose from the patient's own CGM
> signal, through a ridge model trained on 400 patients.

> On 150 held-out patients the fit recovers sensitivity with R2 0.82, median
> error 8.4%, Spearman 0.91 against the simulator's true values.

> Those numbers validate the fitting procedure. They do not validate the
> physiology on real people. That distinction matters.

## 16:15 to 17:45. Live demo, cohort. Slide 18 if the browser fails.

**Do this before you speak:**

1. Open `/` and click **Enter clinic** (top right). You should already be signed
   in; if not, sign in with the demo email, wait for the code, and continue.
2. You land on **Panel review**.
3. Click the **Critical** filter.
4. Leave the sort on **2-hour risk**.
5. Click the row for **GT-0568**.

**Say while clicking:**

> This is the panel. Sixty synthetic patients, sorted by two-hour risk, with a
> status chip. Critical, watch, stable.

> I will take the top critical patient, GT-0568. Fifty years old, seven years
> since diagnosis, HbA1c 7.0, on metformin only.

If asked why the panel is small: it is a 60-patient slice of 1,500, chosen to
show a realistic mix.

## 17:45 to 19:15. Live demo, twin and why.

**On screen after the click:** the twin workspace.

**Do this:**

1. Point at the trace and the shaded band.
2. Click the **LIVE** tab if it is not already active.
3. Point at **Risk** on the right: 100%.
4. Scroll to **Why this alert**.

**Say:**

> Six hours of CGM in black, the next two hours in clay, with an uncertainty
> band. Current reading 127 and falling, but the risk is 100% because the
> evening meal is still ahead.

> The gauge is the calibrated event probability. Time in range is 91%, which
> looks fine until you see where the remaining 9% sits.

> Why this alert: highest glucose in the last three hours, time of day, the
> six-hour average. No sensor gap, no low reading, so it is not a compression
> low. In plain language, this is the dinner window, not a sensor error.

**Optional:** click the play button (title "Replay the day") and let it run for
five seconds, then click **Now**. This shows the live-stream mode.

## 19:15 to 20:30. Live demo, what-if.

**Do this:**

1. Scroll to **What if**.
2. Click **Walk after dinner**.
3. Let the scenario curve draw. Point at **Peak glucose**, **Minutes above 180**
   and **Event probability**.
4. Click **Swap rice for millets**.
5. Click **Skip evening snack** and leave it on screen for two seconds.

**Say:**

> This is the part a clinician can use. Same patient, same evening. A 30-minute
> walk after dinner cuts the peak by about 12 mg/dL and the event probability
> from 38% to 16%.

> Swapping rice for millets cuts the peak by 28 mg/dL. Skipping the snack does
> almost nothing for this patient, so do not spend the conversation there.

> Every scenario uses this patient's fitted parameters. It is a simulation for
> decision support. It is not a prescription, and the screen says so.

**Then:** scroll to **Alert timeline**, point at the filled and hollow dots, and
say:

> Ten alerts in three days, nine matched a real event. The number above each dot
> is the warning lead time.

## 20:30 to 21:15. Slide 19. Limits.

**On screen:** slide 19.

**Say:**

> What we did not prove. This is synthetic data, so no clinical accuracy is
> claimed. External validation is written but has not been run. The deployed
> threshold produces about 2.7 alerts per patient-day, which is too noisy for a
> ward. Subgroup fairness is not measured. And this is not a medical device.

> Under DPDP and HIPAA, a real version needs consent, purpose limitation, audit
> logs and a data-fiduciary agreement. We built with those in mind, but we are
> not claiming compliance.

## 21:15 to 22:00. Slides 20 and 21. Roadmap and close.

**On screen:** slide 20, then slide 21.

**Say:**

> Next is a prospective study in one clinic, with the threshold frozen in
> advance. Then ABDM and ABHA records, Apple Health and Google Fit, and CGM
> vendor APIs. Federated learning across clinics, so records never move. And a
> fairer cohort with CKD, SGLT2 and GLP-1 patients.

**Click:** closing slide.

> GlucoTwin is a decision aid, not a doctor. The clinician decides. Thank you.

Stop the recording. Do not add music or a logo animation.

---

## Two-minute recording checklist

Do this immediately before every take.

1. **Fresh results.** Run `python3 -m pytest -q` and confirm it passes. If you
   changed any model or data code, run `make eval && make export` first, because
   the dashboard reads the exported JSON.
2. **Dashboard loads.** Open `/dashboard`, wait for "Panel review", and confirm
   the patient count is 60. A blank screen usually means the Convex process or
   typecheck failed, not a data problem.
3. **Sign in early.** Use the demo email and complete the OTP before recording.
   Do not record the code arriving.
4. **Pre-open the patient.** Click GT-0568 once, confirm the twin and the five
   scenario buttons render, then go back to the cohort. This warms the JSON
   cache.
5. **Reset state.** Click **Now** if replay is on. Clear any clinician notes you
   do not want on camera. Acknowledge no alerts, or acknowledge them on camera
   deliberately.
6. **Window hygiene.** One browser window, one tab, zoom 100%, bookmarks bar
   hidden, notifications off, no personal email visible in the sidebar.
7. **Terminal, if shown.** Font at 16pt or larger, `pytest` output visible, no
   secrets or `.env` values on screen.
8. **Audio.** Record a 10-second test, play it back, confirm no fan noise and
   that the levels peak around -12 dB.
9. **Timer.** Put a visible clock behind the camera. The live demo must start by
   16:15 and end by 20:30.
10. **Fallback.** Keep `docs/screenshots/` open in a second tab. If the app
    fails during the take, switch to the rendered plates and keep the same
    script. Do not debug on camera.

### Trim plan if you are over 22 minutes

- Cut slide 13 (ablation) to one sentence, saves 45 seconds.
- Cut the calibration plot, keep the Brier number, saves 40 seconds.
- Shorten the data slides, saves 30 seconds.
- Never cut the what-if demo or the limits slide.
