import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/convex/_generated/api";
import { useMutation, useQuery } from "convex/react";
import {
  Activity,
  ArrowLeft,
  Check,
  ChevronRight,
  Footprints,
  HeartPulse,
  Moon,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";
import {
  Area,
  AreaChart,
  ComposedChart,
  CartesianGrid,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { RiskGauge } from "@/components/clinic/RiskGauge";
import { TwinChart } from "@/components/clinic/TwinChart";
import {
  formatClock,
  fortnightSeries,
  glucoseTone,
  nearestCheckpoint,
  pct,
  recentSeries,
  replaySeries,
  statusLabel,
  statusTone,
  trendGlyph,
  type AlertEvent,
  type PatientBundle,
  type SeriesPoint,
  type Status,
} from "@/lib/clinic";

const SCENARIO_ORDER = ["walk_after_dinner", "swap_rice", "skip_snack", "sleep_7h", "metformin_up"];

function riskStatus(risk: number): Status {
  if (risk > 0.6) return "critical";
  if (risk > 0.3) return "watch";
  return "stable";
}

function Panel({
  label,
  meta,
  children,
  className = "",
}: {
  label: string;
  meta?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`border border-border bg-card ${className}`}>
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-2.5">
        <h2 className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{label}</h2>
        {meta ? <div className="text-[11px] text-muted-foreground">{meta}</div> : null}
      </header>
      <div className="px-4 py-4">{children}</div>
    </section>
  );
}

function Sparkline({
  values,
  color,
  height = 46,
  domain,
}: {
  values: number[];
  color: string;
  height?: number;
  domain?: [number, number];
}) {
  const data = values.map((value, index) => ({ index, value }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id={`spark-${color.replace(/[^a-z0-9]/gi, "")}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.22} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <YAxis hide domain={domain ?? ["dataMin - 4", "dataMax + 4"]} />
        <Area
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={1.3}
          fill={`url(#spark-${color.replace(/[^a-z0-9]/gi, "")})`}
          isAnimationActive={false}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function StaticChip({ status }: { status: Status }) {
  const tone = statusTone(status);
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.1em] ${tone.chip} ${tone.text}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
      {statusLabel(status)}
    </span>
  );
}

function Stat({ label, value, unit, tone = "" }: { label: string; value: string; unit?: string; tone?: string }) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
      <div className={`num mt-1 text-[17px] leading-none ${tone}`}>
        {value}
        {unit ? <span className="ml-1 text-[10px] font-normal text-muted-foreground">{unit}</span> : null}
      </div>
    </div>
  );
}

function WhyPanel({ patient, minute }: { patient: PatientBundle; minute: number }) {
  const checkpoint = nearestCheckpoint(patient, minute);
  const factors = (checkpoint?.factors ?? []).slice(0, 5);
  const peak = checkpoint ? Math.max(...checkpoint.forecast.mean) : null;
  const peakAt = checkpoint ? checkpoint.minute + checkpoint.forecast.mean.indexOf(peak as number) * 5 : null;
  const seen = new Set<string>();
  const unique = factors.filter((factor) => {
    if (seen.has(factor.label)) return false;
    seen.add(factor.label);
    return true;
  });
  const max = Math.max(0.0001, ...unique.map((factor) => Math.abs(factor.contribution)));
  const raising = unique.filter((factor) => factor.direction === "raises");
  const lowering = unique.filter((factor) => factor.direction === "lowers");

  return (
    <div className="space-y-3">
      <p className="text-[12px] leading-5 text-muted-foreground">
        {peak !== null ? (
          <>
            Projected to peak near <span className="num text-foreground">{Math.round(peak)} mg/dL</span> around{" "}
            <span className="num text-foreground">{formatClock(peakAt ?? minute)}</span>.
          </>
        ) : (
          "No projection available for this minute."
        )}
      </p>
      <ul className="space-y-2.5">
        {unique.map((factor) => (
          <li key={`${factor.label}-${factor.direction}`}>
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-[12px] leading-4">{factor.label}</span>
              <span
                className={`num text-[10px] ${
                  factor.direction === "raises" ? "text-[#8f3d1c]" : "text-[#4d5a42]"
                }`}
              >
                {factor.direction === "raises" ? "+" : "-"}
                {Math.abs(factor.contribution).toFixed(2)}
              </span>
            </div>
            <div className="mt-1 h-[2px] w-full bg-border">
              <div
                className="h-full"
                style={{
                  width: `${Math.round((Math.abs(factor.contribution) / max) * 100)}%`,
                  backgroundColor: factor.direction === "raises" ? "#b4552d" : "#6f8060",
                }}
              />
            </div>
          </li>
        ))}
      </ul>
      <p className="border-t border-border pt-2.5 text-[11px] leading-5 text-muted-foreground">
        {raising.length > 0 ? (
          <>
            {raising[0].label.charAt(0).toUpperCase() + raising[0].label.slice(1)} is the strongest
            factor pushing the risk up
            {lowering.length > 0 ? `; ${lowering[0].label} is pulling it down` : ""}.
          </>
        ) : (
          "No factor is pushing this window up strongly."
        )}
      </p>
      <p className="text-[10px] leading-4 text-muted-foreground/80">
        Relative contributions from SHAP values on the fused tree model. They explain the model,
        not the physiology.
      </p>
    </div>
  );
}

function RecordRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-border/60 py-1.5 last:border-b-0">
      <dt className="text-[11px] text-muted-foreground">{label}</dt>
      <dd className="num text-[12px]">{value}</dd>
    </div>
  );
}

function RecordPanel({ patient }: { patient: PatientBundle }) {
  const ehr = patient.ehr;
  const meds = [
    ehr.metformin ? "Metformin" : null,
    ehr.sulfonylurea ? "Sulfonylurea" : null,
    ehr.dpp4 ? "DPP-4 inhibitor" : null,
    ehr.insulin ? "Insulin" : null,
  ].filter(Boolean) as string[];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        {meds.length === 0 ? (
          <span className="text-[11px] text-muted-foreground">No glucose-lowering medication recorded</span>
        ) : (
          meds.map((med) => (
            <span key={med} className="border border-border px-2 py-0.5 text-[10px] tracking-wide">
              {med}
            </span>
          ))
        )}
      </div>
      <dl>
        <RecordRow label="HbA1c" value={`${ehr.hba1c.toFixed(1)}%`} />
        <RecordRow label="Fasting glucose" value={`${ehr.fpg_mgdl} mg/dL`} />
        <RecordRow label="eGFR" value={`${ehr.egfr} mL/min/1.73m2`} />
        <RecordRow label="LDL" value={`${ehr.ldl} mg/dL`} />
        <RecordRow label="Triglycerides" value={`${ehr.triglycerides} mg/dL`} />
        <RecordRow label="Systolic BP" value={`${ehr.systolic_bp} mmHg`} />
        <RecordRow label="Waist" value={`${ehr.waist_cm.toFixed(1)} cm`} />
        <RecordRow label="Years since diagnosis" value={`${ehr.years_since_dx.toFixed(1)}`} />
        <RecordRow label="Family history" value={ehr.family_history ? "Yes" : "No"} />
        <RecordRow label="Prior hypoglycaemia" value={`${ehr.prior_hypo_count} events`} />
        <RecordRow label="TCF7L2 rs7903146" value={ehr.tcf7l2} />
        <RecordRow label="Diet" value={ehr.diet_type === "veg" ? "Vegetarian" : "Non-vegetarian"} />
        <RecordRow label="Hypertension" value={ehr.hypertension ? "Yes" : "No"} />
      </dl>
      <p className="text-[10px] leading-4 text-muted-foreground">
        Fitted twin parameters: insulin sensitivity {(patient.params.s_i * 100).toFixed(0)} index,
        basal glucose {Math.round(patient.params.g_basal)} mg/dL, carb absorption{" "}
        {patient.params.tau_g.toFixed(0)} min.
      </p>
    </div>
  );
}

function ScenarioPanel({ patient }: { patient: PatientBundle }) {
  const [selected, setSelected] = useState(SCENARIO_ORDER[0]);
  const scenario = patient.whatif.scenarios[selected];
  const baseline = patient.whatif.baseline;
  const start = patient.whatif.start_min;
  const data = baseline.curve.map((value, index) => ({
    label: start + index * 5,
    baseline: value,
    scenario: scenario ? scenario.curve[index] : null,
  }));

  if (!scenario) return null;
  const betterPeak = scenario.delta_peak < 0;
  const betterMinutes = scenario.delta_minutes_above_180 < 0;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        {SCENARIO_ORDER.map((key) => {
          const option = patient.whatif.scenarios[key];
          if (!option) return null;
          const active = key === selected;
          return (
            <button
              key={key}
              type="button"
              onClick={() => setSelected(key)}
              className={`border px-2.5 py-1.5 text-[11px] transition-colors ${
                active
                  ? "border-foreground/60 bg-foreground text-background"
                  : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground"
              }`}
            >
              {option.label}
            </button>
          );
        })}
      </div>

      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>17:00 to midnight, same fitted parameters</span>
        <span className="flex items-center gap-3">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-px w-4 bg-[#9a9384]" /> current plan
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-px w-4 bg-[#b4552d]" /> scenario
          </span>
        </span>
      </div>

      <ResponsiveContainer width="100%" height={190}>
        <ComposedChart data={data} margin={{ top: 6, right: 10, bottom: 0, left: -8 }}>
          <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="2 5" />
          <XAxis
            dataKey="label"
            tickFormatter={(value: number) => formatClock(value)}
            minTickGap={40}
            axisLine={false}
            tickLine={false}
          />
          <YAxis domain={[60, 320]} ticks={[70, 120, 180, 240, 300]} width={36} axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{
              border: "1px solid var(--border)",
              borderRadius: 2,
              background: "var(--card)",
              fontSize: 11,
              boxShadow: "none",
            }}
            labelFormatter={(value) => formatClock(Number(value))}
            formatter={(value, name) => [
              `${Math.round(Number(value))} mg/dL`,
              name === "baseline" ? "current plan" : "scenario",
            ]}
          />
          <ReferenceLine y={180} stroke="var(--chart-1)" strokeDasharray="2 4" strokeOpacity={0.45} />
          <Line type="monotone" dataKey="baseline" stroke="#9a9384" strokeWidth={1.3} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="scenario" stroke="#b4552d" strokeWidth={1.9} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-3 gap-3 border-t border-border pt-3">
        <Stat
          label="Peak glucose"
          value={`${betterPeak ? "" : "+"}${scenario.delta_peak.toFixed(0)}`}
          unit="mg/dL"
          tone={betterPeak ? "text-[#4d5a42]" : "text-[#8f3d1c]"}
        />
        <Stat
          label="Minutes above 180"
          value={`${betterMinutes ? "" : "+"}${scenario.delta_minutes_above_180}`}
          unit="min"
          tone={betterMinutes ? "text-[#4d5a42]" : "text-[#8f3d1c]"}
        />
        <Stat
          label="Event probability"
          value={`${pct(baseline.prob)} to ${pct(scenario.prob)}`}
          tone={scenario.delta_prob < 0 ? "text-[#4d5a42]" : "text-[#8f3d1c]"}
        />
      </div>
      <p className="text-[10px] leading-4 text-muted-foreground">
        {scenario.description}. Simulation with this patient's own fitted parameters, for decision
        support only. Not medical advice. Sleep changes act on the following morning, so their
        effect falls outside this evening window.
      </p>
    </div>
  );
}

function AlertTimeline({
  patient,
  acks,
  onToggle,
}: {
  patient: PatientBundle;
  acks: string[];
  onToggle: (key: string) => void;
}) {
  const alerts = [...patient.alerts].sort((a, b) => b.day - a.day || b.minute_of_day - a.minute_of_day).slice(0, 8);

  if (alerts.length === 0) {
    return <p className="text-[12px] text-muted-foreground">No alerts in the last 72 hours.</p>;
  }

  return (
    <ol className="divide-y divide-border/70">
      {alerts.map((alert: AlertEvent) => {
        const key = `${alert.day}-${alert.minute_of_day}`;
        const acked = acks.includes(key);
        return (
          <li key={key} className="flex items-start gap-3 py-2.5 first:pt-0 last:pb-0">
            <div className="num w-[70px] shrink-0 text-[11px] text-muted-foreground">
              <div className="text-foreground">{formatClock(alert.minute_of_day)}</div>
              <div>day {alert.day + 1}</div>
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="num text-[11px]">{pct(alert.risk)} risk</span>
                <span className="text-[10px] text-muted-foreground">at {alert.glucose} mg/dL</span>
                {alert.matched && alert.lead_min !== null ? (
                  <span className="border border-[#d9b6a3] bg-[#f7ece5] px-1.5 py-0.5 text-[9px] uppercase tracking-[0.08em] text-[#8f3d1c]">
                    led by {alert.lead_min} min
                  </span>
                ) : (
                  <span className="border border-border px-1.5 py-0.5 text-[9px] uppercase tracking-[0.08em] text-muted-foreground">
                    no event followed
                  </span>
                )}
              </div>
              <p className="mt-1 truncate text-[11px] text-muted-foreground">
                {alert.factors.slice(0, 2).map((factor) => factor.label).join(" / ")}
              </p>
            </div>
            <button
              type="button"
              onClick={() => onToggle(key)}
              title={acked ? "Reopen alert" : "Acknowledge alert"}
              className={`mt-0.5 inline-flex h-6 items-center gap-1 border px-2 text-[10px] transition-colors ${
                acked
                  ? "border-[#c8d1bd] bg-[#eef1e8] text-[#4d5a42]"
                  : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground"
              }`}
            >
              {acked ? <Check className="size-3" /> : null}
              {acked ? "Seen" : "Ack"}
            </button>
          </li>
        );
      })}
    </ol>
  );
}

function NotesPanel({ patientId }: { patientId: string }) {
  const notes = useQuery(api.clinic.listNotes, { patientId });
  const addNote = useMutation(api.clinic.addNote);
  const [body, setBody] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, []);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!body.trim()) return;
    setSaving(true);
    await addNote({ patientId, body });
    setBody("");
    setSaving(false);
    setSaved(true);
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setSaved(false), 2200);
  };

  return (
    <div className="space-y-3">
      <form onSubmit={submit} className="space-y-2">
        <textarea
          value={body}
          onChange={(event) => setBody(event.target.value)}
          rows={2}
          maxLength={400}
          placeholder="Add a note for the next review..."
          className="w-full resize-none border border-border bg-transparent px-2.5 py-2 text-[12px] outline-none placeholder:text-muted-foreground/70 focus:border-foreground/40"
        />
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-muted-foreground">{saved ? "Saved to this clinic account" : "Visible to you only"}</span>
          <button
            type="submit"
            disabled={saving || !body.trim()}
            className="border border-border px-2.5 py-1 text-[10px] uppercase tracking-[0.12em] transition-colors hover:border-foreground/40 disabled:opacity-40"
          >
            {saving ? "Saving" : "Save note"}
          </button>
        </div>
      </form>
      {notes && notes.length > 0 ? (
        <ul className="space-y-2 border-t border-border pt-2.5">
          {notes.slice(0, 3).map((note) => (
            <li key={note.id} className="text-[11px] leading-5">
              <span className="text-muted-foreground">
                {new Date(note.createdAt).toLocaleString(undefined, {
                  day: "2-digit",
                  month: "short",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
              <p className="text-foreground">{note.body}</p>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export function TwinWorkspace({
  patient,
  minute,
  playing,
  onMinute,
  onPlaying,
  onBack,
}: {
  patient: PatientBundle;
  minute: number;
  playing: boolean;
  onMinute: (minute: number) => void;
  onPlaying: (playing: boolean) => void;
  onBack: () => void;
}) {
  const [range, setRange] = useState<"live" | "48h" | "14d">("live");
  const nowMinute = patient.summary.now_minute;
  const clampedMinute = Math.min(minute, nowMinute);
  const replaying = clampedMinute < nowMinute - 5;

  const acks = useQuery(api.clinic.listAcks, { patientId: patient.id }) ?? [];
  const toggleAck = useMutation(api.clinic.toggleAck);

  const checkpoint = nearestCheckpoint(patient, clampedMinute);
  const risk = checkpoint ? checkpoint.risk : patient.summary.risk;
  const status = riskStatus(risk);
  const tone = statusTone(status);

  const day = patient.series.replay_day;
  const upto = Math.min(day.glucose.length - 1, Math.floor(clampedMinute / 5));
  const glucoseNow = day.glucose[upto] ?? patient.summary.glucose;
  const glucose30 = day.glucose[Math.max(0, upto - 6)] ?? glucoseNow;
  const trend = glucoseNow - glucose30;
  const tir = patient.summary.tir;

  const liveSeries = useMemo<SeriesPoint[]>(() => {
    if (replaying) return replaySeries(patient, clampedMinute);
    return recentSeries(patient);
  }, [patient, clampedMinute, replaying]);

  const chartPoints = range === "live" ? liveSeries : range === "48h" ? recentSeries(patient) : fortnightSeries(patient);
  const chartHeight = 300;

  const hrValues = day.hr.slice(0, upto + 1);
  const hrvValues = day.hrv.slice(0, upto + 1).map((value) => value / 10);
  const sleepValues = day.stage.slice(0, upto + 1);
  const stepsToday = day.steps.slice(0, upto + 1).reduce((sum, value) => sum + value, 0);
  const asleepHours = sleepValues.filter((value) => value > 0).length * 5 / 60;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <button
            type="button"
            onClick={onBack}
            className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-[0.16em] text-muted-foreground transition-colors hover:text-foreground"
          >
            <ArrowLeft className="size-3" />
            Cohort
          </button>
          <div className="mt-3 flex flex-wrap items-end gap-3">
            <h1 className="num text-[28px] leading-none">{patient.id}</h1>
            <StaticChip status={status} />
            <span className="text-[11px] text-muted-foreground">
              {patient.ehr.age}y · {patient.ehr.sex.toLowerCase()} · {patient.ehr.diet_type === "veg" ? "vegetarian" : "non-vegetarian"} ·
              TCF7L2 {patient.ehr.tcf7l2}
            </span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-2 border border-border px-2 py-1.5">
            <button
              type="button"
              onClick={() => onPlaying(!playing)}
              className="inline-flex size-6 items-center justify-center border border-border transition-colors hover:border-foreground/40"
              title={playing ? "Pause replay" : "Replay the day"}
            >
              {playing ? <Pause className="size-3" /> : <Play className="size-3" />}
            </button>
            <input
              type="range"
              min={0}
              max={nowMinute}
              step={5}
              value={clampedMinute}
              onChange={(event) => {
                onPlaying(false);
                onMinute(Number(event.target.value));
              }}
              className="h-1 w-32 accent-foreground"
              aria-label="Replay minute of day"
            />
            <span className="num w-10 text-[11px]">{formatClock(clampedMinute)}</span>
            <button
              type="button"
              onClick={() => {
                onPlaying(false);
                onMinute(nowMinute);
              }}
              className="inline-flex items-center gap-1 text-[10px] uppercase tracking-[0.1em] text-muted-foreground transition-colors hover:text-foreground"
              title="Jump to the current reading"
            >
              <RotateCcw className="size-3" />
              Now
            </button>
          </div>
          {replaying ? (
            <span className="border border-[#ddcbab] bg-[#f8f2e6] px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-[#7d5f2c]">
              Replaying {formatClock(clampedMinute)}
            </span>
          ) : (
            <span className="border border-border px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
              Live twin, {formatClock(nowMinute)}
            </span>
          )}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-4">
          <Panel
            label={replaying ? `Glucose, replay to ${formatClock(clampedMinute)}` : "Glucose, last 6 hours and next 2 hours"}
            meta={
              <div className="flex items-center gap-1">
                {(["live", "48h", "14d"] as const).map((option) => (
                  <button
                    key={option}
                    type="button"
                    onClick={() => setRange(option)}
                    className={`px-2 py-1 text-[10px] uppercase tracking-[0.1em] transition-colors ${
                      range === option ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {option}
                  </button>
                ))}
              </div>
            }
          >
            <TwinChart
              points={chartPoints}
              nowLabel={range === "live" ? clampedMinute : range === "48h" ? nowMinute : undefined}
              height={chartHeight}
              xKind={range === "14d" ? "day" : "clock"}
            />
            <div className="mt-4 grid grid-cols-2 gap-4 border-t border-border pt-3 sm:grid-cols-4">
              <Stat label="Glucose now" value={`${Math.round(glucoseNow)}`} unit="mg/dL" tone={glucoseTone(glucoseNow)} />
              <Stat
                label="30-minute trend"
                value={`${trend > 0 ? "+" : ""}${Math.round(trend)}`}
                unit="mg/dL"
                tone={trend > 0 ? "text-[#8f3d1c]" : trend < 0 ? "text-[#4d5a42]" : ""}
              />
              <Stat label="Time in range" value={pct(tir.in_range)} unit="24 h" />
              <Stat label="Alerts" value={`${patient.summary.alerts_72h}`} unit="72 h" />
            </div>
            <p className="mt-2 text-[10px] text-muted-foreground">
              {replaying
                ? "Solid line is observed CGM. Shaded band is the model projection at this minute."
                : "Solid line is observed CGM. The shaded band is the next two hours, 80% interval."}{" "}
              Trend is {trendGlyph(trend)}.
            </p>
          </Panel>

          <div className="grid gap-4 sm:grid-cols-3">
            <Panel label="Heart rate" meta={<HeartPulse className="size-3.5 text-muted-foreground" />}>
              <Sparkline values={hrValues.slice(-72)} color="#b4552d" />
              <div className="mt-2 flex items-baseline justify-between">
                <span className="num text-[17px]">{Math.round(hrValues[hrValues.length - 1] ?? 0)}</span>
                <span className="text-[10px] text-muted-foreground">bpm</span>
              </div>
            </Panel>
            <Panel label="HRV (RMSSD)" meta={<Activity className="size-3.5 text-muted-foreground" />}>
              <Sparkline values={hrvValues.slice(-72)} color="#6f8060" />
              <div className="mt-2 flex items-baseline justify-between">
                <span className="num text-[17px]">{(hrvValues[hrvValues.length - 1] ?? 0).toFixed(0)}</span>
                <span className="text-[10px] text-muted-foreground">ms</span>
              </div>
            </Panel>
            <Panel label="Sleep and steps" meta={<Moon className="size-3.5 text-muted-foreground" />}>
              <Sparkline values={sleepValues.slice(-72)} color="#7a8391" domain={[0, 3]} />
              <div className="mt-2 flex items-baseline justify-between text-[10px] text-muted-foreground">
                <span className="num text-[17px] text-foreground">{asleepHours.toFixed(1)}<span className="ml-1 text-[10px] font-normal text-muted-foreground">h asleep</span></span>
                <span className="inline-flex items-center gap-1">
                  <Footprints className="size-3" />
                  <span className="num text-[12px] text-foreground">{stepsToday.toLocaleString()}</span>
                </span>
              </div>
            </Panel>
          </div>

          <Panel
            label="What if"
            meta={<span className="text-[10px] uppercase tracking-[0.12em]">counterfactual, same patient</span>}
          >
            <ScenarioPanel patient={patient} />
          </Panel>

          <Panel label="Alert timeline" meta={<span>last 72 hours</span>}>
            <AlertTimeline
              patient={patient}
              acks={acks}
              onToggle={(key) => {
                void toggleAck({ patientId: patient.id, alertKey: key });
              }}
            />
          </Panel>
        </div>

        <aside className="space-y-4">
          <Panel label="Risk" meta={<StaticChip status={status} />}>
            <RiskGauge risk={risk} status={status} />
            <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 border-t border-border pt-3">
              <Stat label="Highest in 24 h" value={pct(patient.summary.risk_max_24h)} />
              <Stat label="Projection peak" value={checkpoint ? `${Math.max(...checkpoint.forecast.mean)}` : "-"} unit="mg/dL" />
              <Stat label="Above 180" value={pct(tir.above)} unit="of 24 h" />
              <Stat label="Below 70" value={pct(tir.below)} unit="of 24 h" />
            </div>
            <p className={`mt-3 border-t border-border pt-2.5 text-[11px] leading-5 ${tone.text}`}>
              {status === "critical"
                ? "High probability of a hyperglycaemic event in the next two hours."
                : status === "watch"
                  ? "Borderline. Worth a look before the next meal."
                  : "Low probability of a hyperglycaemic event in the next two hours."}
            </p>
          </Panel>

          <Panel label="Why this alert">
            <WhyPanel patient={patient} minute={clampedMinute} />
          </Panel>

          <Panel label="Record">
            <RecordPanel patient={patient} />
          </Panel>

          <Panel label="Clinician notes" meta={<span className="text-[10px]">saved to Convex</span>}>
            <NotesPanel patientId={patient.id} />
          </Panel>
        </aside>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4 text-[10px] text-muted-foreground">
        <span>
          Synthetic patient, generated by this repository. Every value is reproducible from a fixed seed.
        </span>
        <span className="inline-flex items-center gap-1.5">
          Not medical advice
          <ChevronRight className="size-3" />
          decision support only
        </span>
      </div>
    </div>
  );
}
