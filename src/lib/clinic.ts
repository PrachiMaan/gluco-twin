/**
 * Data access for the exported GlucoTwin artifacts.
 *
 * Everything is produced by `python src/export_web.py` from the trained models.
 * The browser only reads these files; it never invents a number.
 */

export type Status = "critical" | "watch" | "stable";

export interface PanelPatient {
  id: string;
  age: number;
  sex: string;
  hba1c: number;
  bmi: number;
  risk: number;
  status: Status;
  glucose: number;
  trend_30m: number;
  tir_in_range: number;
  alerts_72h: number;
}

export interface CohortPayload {
  generated_at: string;
  cohort: {
    n_patients: number;
    n_days: number;
    step_min: number;
    panel_size: number;
    mean_hba1c: number;
    mean_age: number;
    mean_bmi: number;
    synthetic: boolean;
  };
  patients: PanelPatient[];
}

export interface Ehr {
  age: number;
  sex: string;
  bmi: number;
  waist_cm: number;
  diet_type: string;
  family_history: number;
  years_since_dx: number;
  hba1c: number;
  fpg_mgdl: number;
  egfr: number;
  ldl: number;
  triglycerides: number;
  systolic_bp: number;
  metformin: number;
  sulfonylurea: number;
  dpp4: number;
  insulin: number;
  hypertension: number;
  tcf7l2: string;
  prior_hypo_count: number;
}

export interface Factor {
  label: string;
  feature?: string;
  value?: number;
  contribution: number;
  direction: "raises" | "lowers";
}

export interface Forecast {
  mean: number[];
  lo: number[];
  hi: number[];
}

export interface ReplayCheckpoint {
  minute: number;
  risk: number;
  glucose: number;
  forecast: Forecast;
  factors: Factor[];
}

export interface AlertEvent {
  minute_of_day: number;
  day: number;
  risk: number;
  lead_min: number | null;
  glucose: number;
  matched: boolean;
  factors: { label: string; contribution: number; direction: "raises" | "lowers" }[];
}

export interface ScenarioResult {
  label: string;
  description: string;
  curve: number[];
  peak: number;
  minutes_above_180: number;
  prob: number;
  delta_prob: number;
  delta_peak: number;
  delta_minutes_above_180: number;
}

export interface WhatIf {
  start_min: number;
  checkpoint_min: number;
  minutes: number[];
  baseline: {
    curve: number[];
    peak: number;
    minutes_above_180: number;
    prob: number;
    risk_profile: number[];
  };
  scenarios: Record<string, ScenarioResult>;
}

export interface PatientBundle {
  id: string;
  index: number;
  ehr: Ehr;
  params: { g_basal: number; s_i: number; tau_g: number; x_gain: number };
  profile: Record<string, number>;
  summary: {
    risk: number;
    status: Status;
    glucose: number;
    trend_30m: number;
    now_minute: number;
    tir: { in_range: number; above: number; below: number; mean: number; hours: number; n: number };
    risk_max_24h: number;
    alerts_72h: number;
  };
  series: {
    glucose_14d: { step_min: number; values: number[] };
    recent: { hours: number; step_min: number; values: number[] };
    replay_day: {
      day: number;
      step_min: number;
      glucose: number[];
      hr: number[];
      hrv: number[];
      steps: number[];
      stage: number[];
    };
    wearables_daily: {
      day: number;
      steps: number;
      hr_mean: number;
      hrv_mean: number;
      sleep_h: number;
      glucose_mean: number;
    }[];
  };
  risk_trace: { step_min: number; values: number[] };
  replay_checkpoints: ReplayCheckpoint[];
  alerts: AlertEvent[];
  whatif: WhatIf;
}

export interface AblationRow {
  model: string;
  auroc: number;
  auprc: number;
  within_patient_auroc: number;
  brier: number;
  sensitivity_at_far: number;
  false_alarms_per_day: number;
  mean_lead_min: number | null;
  threshold: number;
}

export interface MetricsPayload {
  test_patients: number;
  test_windows: number;
  event_base_rate: number;
  deployed_model: string;
  models: Record<
    string,
    {
      auroc: number;
      auprc: number;
      within_patient_auroc: number;
      brier: number;
      sensitivity?: number;
      false_alarms_per_day?: number;
      mean_lead_min?: number;
      threshold?: number;
      n_events?: number;
    }
  >;
  dense_deployed?: {
    auroc: number;
    auprc: number;
    brier: number;
    sensitivity: number;
    false_alarms_per_day: number;
    mean_lead_min: number;
    n_events: number;
    cadence_min: number;
  };
  forecast: Record<string, { rmse: number; mae: number }>;
  ablation: AblationRow[];
  param_recovery?: {
    s_i_log_r2: number;
    s_i_median_abs_pct_error: number;
    s_i_spearman: number;
    n_train: number;
    n_val: number;
  };
  top_features?: { feature: string; mean_abs_shap: number }[];
}

const patientCache = new Map<string, Promise<PatientBundle>>();
let cohortCache: Promise<CohortPayload> | null = null;
let metricsCache: Promise<MetricsPayload> | null = null;

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path} (${response.status})`);
  }
  return (await response.json()) as T;
}

export function loadCohort(): Promise<CohortPayload> {
  if (!cohortCache) cohortCache = getJson<CohortPayload>("/data/cohort.json");
  return cohortCache;
}

export function loadMetrics(): Promise<MetricsPayload> {
  if (!metricsCache) metricsCache = getJson<MetricsPayload>("/data/metrics.json");
  return metricsCache;
}

export function loadPatient(id: string): Promise<PatientBundle> {
  const cached = patientCache.get(id);
  if (cached) return cached;
  const promise = getJson<PatientBundle>(`/data/patients/${id}.json`);
  patientCache.set(id, promise);
  return promise;
}

export function statusLabel(status: Status): string {
  if (status === "critical") return "Critical";
  if (status === "watch") return "Watch";
  return "Stable";
}

export function statusTone(status: Status): {
  dot: string;
  text: string;
  chip: string;
} {
  if (status === "critical") {
    return { dot: "bg-[#b4552d]", text: "text-[#8f3d1c]", chip: "border-[#d9b6a3] bg-[#f7ece5]" };
  }
  if (status === "watch") {
    return { dot: "bg-[#b08a4f]", text: "text-[#7d5f2c]", chip: "border-[#ddcbab] bg-[#f8f2e6]" };
  }
  return { dot: "bg-[#6f8060]", text: "text-[#4d5a42]", chip: "border-[#c8d1bd] bg-[#eef1e8]" };
}

export function formatClock(minutes: number): string {
  const m = ((minutes % 1440) + 1440) % 1440;
  const h = Math.floor(m / 60);
  const mm = Math.round(m % 60);
  return `${String(h).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}

export function formatGlucose(value: number): string {
  return `${Math.round(value)}`;
}

export function glucoseTone(value: number): string {
  if (value > 180) return "text-[#b4552d]";
  if (value < 70) return "text-[#4f6b7a]";
  return "text-foreground";
}

export function trendGlyph(trend: number): string {
  if (trend > 4) return "rising";
  if (trend < -4) return "falling";
  return "steady";
}

export function pct(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`;
}

/** Build a normalised time series for charts. */
export interface SeriesPoint {
  label: number;
  glucose: number | null;
  lo: number | null;
  hi: number | null;
  projected: boolean;
}

export function recentSeries(patient: PatientBundle): SeriesPoint[] {
  const points: SeriesPoint[] = [];
  const nowMinute = patient.summary.now_minute;
  const recent = patient.series.recent.values;
  const startMinute = nowMinute - recent.length * 5;
  recent.forEach((value, i) => {
    points.push({
      label: startMinute + i * 5,
      glucose: value,
      lo: null,
      hi: null,
      projected: false,
    });
  });
  const last = patient.replay_checkpoints[patient.replay_checkpoints.length - 1];
  if (last) {
    // anchor the band at the current reading so the area joins the trace
    points.push({
      label: nowMinute,
      glucose: null,
      lo: last.forecast.mean[0],
      hi: last.forecast.mean[0],
      projected: true,
    });
    last.forecast.mean.forEach((value, i) => {
      if (i === 0) return;
      points.push({
        label: nowMinute + i * 5,
        glucose: null,
        lo: last.forecast.lo[i],
        hi: last.forecast.hi[i],
        projected: true,
      });
    });
  }
  return points;
}

export function fortnightSeries(patient: PatientBundle): SeriesPoint[] {
  const values = patient.series.glucose_14d.values;
  return values.map((value, i) => ({
    label: i * 15,
    glucose: value,
    lo: null,
    hi: null,
    projected: false,
  }));
}

export function replaySeries(patient: PatientBundle, minute: number): SeriesPoint[] {
  const day = patient.series.replay_day;
  const cutoff = Math.min(Math.floor(minute / 5), day.glucose.length - 1);
  const points: SeriesPoint[] = [];
  for (let i = 0; i <= cutoff; i += 1) {
    points.push({ label: i * 5, glucose: day.glucose[i], lo: null, hi: null, projected: false });
  }
  const checkpoint = nearestCheckpoint(patient, minute);
  if (checkpoint) {
    checkpoint.forecast.mean.forEach((value, i) => {
      if (i === 0) return;
      points.push({
        label: minute + i * 5,
        glucose: null,
        lo: checkpoint.forecast.lo[i],
        hi: checkpoint.forecast.hi[i],
        projected: true,
      });
    });
  }
  return points;
}

export function nearestCheckpoint(patient: PatientBundle, minute: number): ReplayCheckpoint | null {
  let best: ReplayCheckpoint | null = null;
  let bestGap = Infinity;
  for (const checkpoint of patient.replay_checkpoints) {
    const gap = Math.abs(checkpoint.minute - minute);
    if (gap < bestGap) {
      bestGap = gap;
      best = checkpoint;
    }
  }
  return best;
}

export function currentFactors(patient: PatientBundle, minute: number): Factor[] {
  const checkpoint = nearestCheckpoint(patient, minute);
  if (checkpoint) return checkpoint.factors;
  return [];
}

export function currentRisk(patient: PatientBundle, minute: number): number {
  if (Math.abs(minute - patient.summary.now_minute) <= 10) return patient.summary.risk;
  const checkpoint = nearestCheckpoint(patient, minute);
  if (checkpoint) return checkpoint.risk;
  return patient.summary.risk;
}

export function replayCheckpoint(patient: PatientBundle, minute: number): ReplayCheckpoint | null {
  let exact: ReplayCheckpoint | null = null;
  let best: ReplayCheckpoint | null = null;
  let bestGap = Infinity;
  for (const checkpoint of patient.replay_checkpoints) {
    const gap = Math.abs(checkpoint.minute - minute);
    if (gap === 0) exact = checkpoint;
    if (gap < bestGap) {
      bestGap = gap;
      best = checkpoint;
    }
  }
  return exact ?? best;
}
