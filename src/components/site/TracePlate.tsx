import { motion } from "framer-motion";
import { useMemo } from "react";

import type { PatientBundle } from "@/lib/clinic";

const W = 680;
const H = 300;
const Y_MIN = 40;
const Y_MAX = 300;

function buildPath(values: number[], x: (i: number, n: number) => number, y: (v: number) => number) {
  return values
    .map((value, i) => `${i === 0 ? "M" : "L"}${x(i, values.length).toFixed(1)},${y(value).toFixed(1)}`)
    .join(" ");
}

export function TracePlate({ patient }: { patient: PatientBundle | null }) {
  const view = useMemo(() => {
    if (!patient) return null;
    const recent = patient.series.recent.values;
    const last = patient.replay_checkpoints[patient.replay_checkpoints.length - 1];
    const forecast = last?.forecast;
    const forecastMean = forecast ? forecast.mean.slice(1) : [];
    const values = [...recent, ...forecastMean];
    const top = Math.max(Y_MAX, ...values);
    const x = (i: number, n: number) => (i / Math.max(n - 1, 1)) * W;
    const y = (v: number) => H - ((v - Y_MIN) / (top - Y_MIN)) * H;
    const nowX = x(recent.length - 1, recent.length + forecastMean.length);

    const trace = buildPath(recent, x, y);
    const projection =
      forecastMean.length > 0
        ? `M${nowX.toFixed(1)},${y(recent[recent.length - 1]).toFixed(1)} ` +
          forecastMean
            .map((value, i) => `L${x(recent.length + i, recent.length + forecastMean.length).toFixed(1)},${y(value).toFixed(1)}`)
            .join(" ")
        : "";

    const band =
      forecast && forecastMean.length > 0
        ? [
            `M${nowX.toFixed(1)},${y(forecast.lo[0]).toFixed(1)}`,
            ...forecast.lo
              .slice(1)
              .map(
                (value, i) =>
                  `L${x(recent.length + i, recent.length + forecastMean.length).toFixed(1)},${y(value).toFixed(1)}`,
              ),
            ...forecast.hi
              .slice(1)
              .reverse()
              .map(
                (value, i) =>
                  `L${x(recent.length + forecast.hi.length - 2 - i, recent.length + forecastMean.length).toFixed(1)},${y(value).toFixed(1)}`,
              ),
            "Z",
          ].join(" ")
        : "";

    return {
      trace,
      projection,
      band,
      nowX,
      thresholdY: y(180),
      lowY: y(70),
      top,
      glucose: recent[recent.length - 1],
      samples: recent.length,
      hours: patient.series.recent.hours,
    };
  }, [patient]);

  return (
    <figure className="relative">
      <div className="border border-border bg-card p-2">
        <div className="relative border border-border/70 paper">
          <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label="CGM trace with projection">
            {view ? (
              <>
                <defs>
                  <linearGradient id="trace-fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--chart-1)" stopOpacity="0.16" />
                    <stop offset="100%" stopColor="var(--chart-1)" stopOpacity="0.01" />
                  </linearGradient>
                </defs>
                <line x1={0} x2={W} y1={view.thresholdY} y2={view.thresholdY} stroke="var(--chart-1)" strokeOpacity="0.35" strokeDasharray="2 4" />
                <line x1={0} x2={W} y1={view.lowY} y2={view.lowY} stroke="var(--chart-3)" strokeOpacity="0.25" strokeDasharray="2 4" />
                <text x={6} y={view.thresholdY - 6} className="fill-[var(--chart-1)] font-mono" fontSize="9" opacity="0.8">
                  180 mg/dL
                </text>
                {view.band ? <path d={view.band} fill="var(--chart-3)" fillOpacity="0.1" stroke="none" /> : null}
                <motion.path
                  d={view.trace}
                  fill="none"
                  stroke="var(--chart-1)"
                  strokeWidth="1.8"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 2.2, ease: "easeInOut" }}
                />
                <line x1={view.nowX} x2={view.nowX} y1={12} y2={H} stroke="var(--foreground)" strokeOpacity="0.25" strokeDasharray="1 3" />
                {view.projection ? (
                  <motion.path
                    d={view.projection}
                    fill="none"
                    stroke="var(--chart-3)"
                    strokeWidth="1.6"
                    strokeDasharray="4 3"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1.4, delay: 1.8, ease: "easeInOut" }}
                  />
                ) : null}
                <text x={view.nowX - 4} y={16} textAnchor="end" className="fill-foreground/50 font-mono" fontSize="9">
                  now
                </text>
              </>
            ) : (
              <rect width={W} height={H} fill="transparent" />
            )}
          </svg>
          {!view ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="label">loading trace</span>
            </div>
          ) : null}
        </div>
      </div>
      <figcaption className="mt-3 flex items-baseline justify-between gap-4 border-t border-border pt-3">
        <span className="label">Plate 01 - one synthetic patient</span>
        <span className="num text-[11px] text-muted-foreground">
          {view ? `${view.hours} h at 5 min - ${view.glucose} mg/dL now` : "48 h at 5 min"}
        </span>
      </figcaption>
    </figure>
  );
}
