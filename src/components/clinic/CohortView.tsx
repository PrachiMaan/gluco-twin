import { useMemo, useState } from "react";
import { Search, ChevronDown } from "lucide-react";

import {
  type CohortPayload,
  type PanelPatient,
  type Status,
  glucoseTone,
  pct,
  statusLabel,
  statusTone,
  trendGlyph,
} from "@/lib/clinic";

type SortKey = "risk" | "glucose" | "hba1c" | "alerts";

const FILTERS: { key: Status | "all"; label: string }[] = [
  { key: "all", label: "All" },
  { key: "critical", label: "Critical" },
  { key: "watch", label: "Watch" },
  { key: "stable", label: "Stable" },
];

const SORTS: { key: SortKey; label: string }[] = [
  { key: "risk", label: "2-hour risk" },
  { key: "glucose", label: "Glucose" },
  { key: "hba1c", label: "HbA1c" },
  { key: "alerts", label: "Alerts" },
];

function TrendMark({ trend }: { trend: number }) {
  const word = trendGlyph(trend);
  const glyph = word === "rising" ? "\u2197" : word === "falling" ? "\u2198" : "\u2192";
  return (
    <span className="ml-1.5 inline-flex items-center gap-1 text-[10px] text-muted-foreground">
      <span aria-hidden>{glyph}</span>
      {word}
    </span>
  );
}

function RiskCell({ patient }: { patient: PanelPatient }) {
  const tone = statusTone(patient.status);
  return (
    <div className="flex items-center gap-2.5">
      <div className="h-[3px] w-16 overflow-hidden bg-border">
        <div
          className={`h-full ${tone.dot}`}
          style={{ width: `${Math.max(2, Math.round(patient.risk * 100))}%` }}
        />
      </div>
      <span className="num text-[13px]">{pct(patient.risk)}</span>
    </div>
  );
}

export function CohortView({
  cohort,
  selectedId,
  onSelect,
}: {
  cohort: CohortPayload;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const [filter, setFilter] = useState<Status | "all">("all");
  const [sort, setSort] = useState<SortKey>("risk");
  const [query, setQuery] = useState("");

  const counts = useMemo(() => {
    const base: Record<string, number> = { all: cohort.patients.length, critical: 0, watch: 0, stable: 0 };
    for (const patient of cohort.patients) base[patient.status] += 1;
    return base;
  }, [cohort]);

  const rows = useMemo(() => {
    const term = query.trim().toLowerCase();
    const list = cohort.patients.filter((patient) => {
      if (filter !== "all" && patient.status !== filter) return false;
      if (!term) return true;
      return (
        patient.id.toLowerCase().includes(term) ||
        patient.sex.toLowerCase().includes(term) ||
        String(Math.round(patient.hba1c)).includes(term)
      );
    });
    const key = sort;
    return [...list].sort((a, b) => {
      if (key === "alerts") return b.alerts_72h - a.alerts_72h;
      if (key === "glucose") return b.glucose - a.glucose;
      if (key === "hba1c") return b.hba1c - a.hba1c;
      return b.risk - a.risk;
    });
  }, [cohort, filter, sort, query]);

  const stats = cohort.cohort;

  return (
    <div className="space-y-7">
      <header className="flex flex-wrap items-end justify-between gap-6">
        <div>
          <div className="eyebrow">Cohort</div>
          <h1 className="mt-2 font-serif text-[34px] leading-tight">Panel review</h1>
          <p className="mt-2 max-w-xl text-[13px] leading-relaxed text-muted-foreground">
            {stats.n_patients} patients with {stats.n_days} days of CGM and wearable data each.
            Sorted by the probability of a hyperglycemic event in the next two hours.
          </p>
        </div>
        <dl className="grid grid-cols-2 gap-x-8 gap-y-3 text-[11px] sm:grid-cols-4">
          {[
            ["Patients", String(stats.n_patients)],
            ["Mean HbA1c", `${stats.mean_hba1c.toFixed(1)}%`],
            ["Mean age", `${Math.round(stats.mean_age)}`],
            ["Mean BMI", stats.mean_bmi.toFixed(1)],
          ].map(([label, value]) => (
            <div key={label}>
              <dt className="uppercase tracking-[0.14em] text-muted-foreground">{label}</dt>
              <dd className="num mt-1 text-[17px]">{value}</dd>
            </div>
          ))}
        </dl>
      </header>

      <div className="flex flex-wrap items-center justify-between gap-4 border-y border-border py-3">
        <div className="flex flex-wrap items-center gap-1.5">
          {FILTERS.map((option) => {
            const active = filter === option.key;
            return (
              <button
                key={option.key}
                type="button"
                onClick={() => setFilter(option.key)}
                className={`rounded-sm border px-3 py-1.5 text-[11px] tracking-wide transition-colors ${
                  active
                    ? "border-foreground/70 bg-foreground text-background"
                    : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground"
                }`}
              >
                {option.label}
                <span className={`num ml-1.5 ${active ? "text-background/60" : "text-muted-foreground/70"}`}>
                  {counts[option.key]}
                </span>
              </button>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Patient ID or age"
              className="h-8 w-44 border border-border bg-transparent pl-8 pr-2 text-[12px] outline-none placeholder:text-muted-foreground/70 focus:border-foreground/40"
            />
          </label>
          <label className="relative">
            <select
              value={sort}
              onChange={(event) => setSort(event.target.value as SortKey)}
              className="h-8 appearance-none border border-border bg-transparent pl-3 pr-8 text-[12px] outline-none focus:border-foreground/40"
            >
              {SORTS.map((option) => (
                <option key={option.key} value={option.key}>
                  Sort: {option.label}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          </label>
        </div>
      </div>

      {/* desktop table */}
      <div className="hidden overflow-x-auto md:block">
        <table className="w-full min-w-[860px] border-collapse text-left">
          <thead>
            <tr className="border-b border-border text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
              <th className="py-2.5 pr-4 font-normal">Patient</th>
              <th className="py-2.5 pr-4 font-normal">Status</th>
              <th className="py-2.5 pr-4 font-normal">2-hour risk</th>
              <th className="py-2.5 pr-4 font-normal">Glucose</th>
              <th className="py-2.5 pr-4 font-normal">Time in range</th>
              <th className="py-2.5 pr-4 font-normal">HbA1c</th>
              <th className="py-2.5 pr-4 font-normal">Alerts / 72h</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((patient) => {
              const tone = statusTone(patient.status);
              const selected = patient.id === selectedId;
              return (
                <tr
                  key={patient.id}
                  onClick={() => onSelect(patient.id)}
                  className={`cursor-pointer border-b border-border/70 transition-colors ${
                    selected ? "bg-accent/70" : "hover:bg-accent/40"
                  }`}
                >
                  <td className="py-3.5 pr-4">
                    <div className="num text-[13px]">{patient.id}</div>
                    <div className="mt-0.5 text-[11px] text-muted-foreground">
                      {patient.age}y · {patient.sex.toLowerCase()} · BMI {patient.bmi.toFixed(1)}
                    </div>
                  </td>
                  <td className="py-3.5 pr-4">
                    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.1em] ${tone.chip} ${tone.text}`}>
                      <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
                      {statusLabel(patient.status)}
                    </span>
                  </td>
                  <td className="py-3.5 pr-4">
                    <RiskCell patient={patient} />
                  </td>
                  <td className="py-3.5 pr-4">
                    <span className={`num text-[13px] ${glucoseTone(patient.glucose)}`}>
                      {Math.round(patient.glucose)}
                    </span>
                    <span className="ml-1 text-[10px] text-muted-foreground">mg/dL</span>
                    <TrendMark trend={patient.trend_30m} />
                  </td>
                  <td className="py-3.5 pr-4">
                    <div className="num text-[12px]">{pct(patient.tir_in_range)}</div>
                    <div className="mt-1 h-[2px] w-20 bg-border">
                      <div className="h-full bg-[#6f8060]" style={{ width: `${Math.round(patient.tir_in_range * 100)}%` }} />
                    </div>
                  </td>
                  <td className="num py-3.5 pr-4 text-[13px]">{patient.hba1c.toFixed(1)}%</td>
                  <td className="num py-3.5 pr-4 text-[13px] text-muted-foreground">{patient.alerts_72h}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* mobile cards */}
      <div className="grid gap-2 md:hidden">
        {rows.map((patient) => {
          const tone = statusTone(patient.status);
          const selected = patient.id === selectedId;
          return (
            <button
              key={patient.id}
              type="button"
              onClick={() => onSelect(patient.id)}
              className={`border p-3 text-left transition-colors ${
                selected ? "border-foreground/40 bg-accent/60" : "border-border hover:bg-accent/40"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="num text-[13px]">{patient.id}</span>
                <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-[0.1em] ${tone.chip} ${tone.text}`}>
                  <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
                  {statusLabel(patient.status)}
                </span>
              </div>
              <div className="mt-2 flex items-end justify-between">
                <div className="text-[11px] text-muted-foreground">
                  {patient.age}y · {patient.sex.toLowerCase()} · A1c {patient.hba1c.toFixed(1)}%
                </div>
                <div className="num text-[13px]">{pct(patient.risk)}</div>
              </div>
            </button>
          );
        })}
      </div>

      {rows.length === 0 ? (
        <p className="border border-dashed border-border p-8 text-center text-[12px] text-muted-foreground">
          No patients match this filter.
        </p>
      ) : null}

      <p className="text-[11px] text-muted-foreground">
        Panel exported {new Date(cohort.generated_at).toLocaleString()} · synthetic data · not for clinical use.
      </p>
    </div>
  );
}
