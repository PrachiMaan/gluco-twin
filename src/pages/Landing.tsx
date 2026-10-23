import { motion } from "framer-motion";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router";

import { TracePlate } from "@/components/site/TracePlate";
import { useAuth } from "@/hooks/use-auth";
import {
  loadCohort,
  loadMetrics,
  loadPatient,
  type CohortPayload,
  type MetricsPayload,
  type PatientBundle,
} from "@/lib/clinic";

const fadeUp = {
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-80px" },
  transition: { duration: 0.6, ease: "easeOut" as const },
};

const MODEL_LABELS: Record<string, string> = {
  persistence: "Persistence baseline",
  lgbm_cgm: "CGM only",
  lgbm_ehr: "Record only",
  lgbm_fused: "Fused (shipped)",
  fusion_net: "Fusion net (TCN + MLP)",
};

function InkButton({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <Link
      to={to}
      className="group inline-flex items-center gap-3 bg-primary px-5 py-3.5 text-[11px] uppercase tracking-[0.18em] text-primary-foreground transition hover:bg-primary/92"
    >
      {children}
      <ArrowRight className="size-3.5 transition group-hover:translate-x-0.5" />
    </Link>
  );
}

export default function Landing() {
  const { isAuthenticated } = useAuth();
  const [cohort, setCohort] = useState<CohortPayload | null>(null);
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [patient, setPatient] = useState<PatientBundle | null>(null);

  useEffect(() => {
    let alive = true;
    loadCohort()
      .then((payload) => {
        if (!alive) return;
        setCohort(payload);
        const first = payload.patients[0];
        if (first) {
          loadPatient(first.id).then((bundle) => alive && setPatient(bundle)).catch(() => undefined);
        }
      })
      .catch(() => undefined);
    loadMetrics().then((payload) => alive && setMetrics(payload)).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  const deployed = metrics?.dense_deployed;
  const rows = metrics?.ablation ?? [];

  return (
    <div className="min-h-screen bg-background">
      {/* nav */}
      <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur">
        <div className="mx-auto flex h-[76px] max-w-6xl items-center justify-between px-6">
          <Link to="/" className="group flex items-baseline gap-2">
            <span className="font-display text-[26px] leading-none tracking-tight">GlucoTwin</span>
            <span className="size-1 rounded-full bg-[#b4552d]" aria-hidden />
          </Link>
          <nav className="hidden items-center gap-8 text-[11px] uppercase tracking-[0.18em] text-muted-foreground md:flex">
            <a href="#problem" className="transition hover:text-foreground">Problem</a>
            <a href="#method" className="transition hover:text-foreground">Method</a>
            <a href="#results" className="transition hover:text-foreground">Results</a>
            <a href="#clinic" className="transition hover:text-foreground">The clinic</a>
          </nav>
          <div className="flex items-center gap-5">
            {isAuthenticated ? (
              <Link to="/dashboard" className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground transition hover:text-foreground">
                Dashboard
              </Link>
            ) : (
              <Link to="/auth" className="hidden text-[11px] uppercase tracking-[0.18em] text-muted-foreground transition hover:text-foreground sm:block">
                Sign in
              </Link>
            )}
            <Link
              to="/dashboard"
              className="inline-flex items-center gap-2 border border-border px-4 py-2.5 text-[11px] uppercase tracking-[0.18em] transition hover:border-foreground/40"
            >
              Enter clinic
              <ArrowUpRight className="size-3" />
            </Link>
          </div>
        </div>
      </header>

      {/* hero */}
      <section className="relative overflow-hidden">
        <div className="mx-auto grid max-w-6xl gap-16 px-6 pb-20 pt-16 md:grid-cols-[1.02fr_0.98fr] md:items-center md:pb-28 md:pt-24">
          <motion.div initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: "easeOut" }}>
            <div className="mb-7 flex items-center gap-3">
              <span className="h-px w-8 bg-foreground/30" />
              <span className="label">Happiest Health - Summit 2026 - proof of concept</span>
            </div>
            <h1 className="font-display text-[46px] leading-[1.02] tracking-tight sm:text-[58px]">
              A working replica of one patient,
              <span className="italic text-[#8f3d1c]"> two hours ahead.</span>
            </h1>
            <p className="mt-7 max-w-[30rem] text-[15px] leading-7 text-muted-foreground">
              GlucoTwin fuses a type 2 diabetes record with 14 days of continuous glucose
              monitoring and wearable data, then keeps a live twin: a 2-hour hyperglycaemia
              forecast, the reasons behind the alert, and a simulator for what happens if the
              plan changes.
            </p>
            <div className="mt-9 flex flex-wrap items-center gap-6">
              <InkButton to={isAuthenticated ? "/dashboard" : "/auth?returnTo=%2Fdashboard"}>
                Open the clinician dashboard
              </InkButton>
              <a href="#method" className="group inline-flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-foreground/70 transition hover:text-foreground">
                Read the method
                <ArrowRight className="size-3 transition group-hover:translate-x-0.5" />
              </a>
            </div>
            <p className="mt-8 text-[12px] leading-5 text-muted-foreground/80">
              Synthetic patients only. No real records. Every number on this page is produced by a
              script in the repository.
            </p>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 26 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.15, ease: "easeOut" }}>
            <TracePlate patient={patient} />
            <div className="mt-2 flex items-center justify-between text-[11px] text-muted-foreground">
              <span>Solid: observed CGM</span>
              <span>Dashed: twin projection, shaded 80% band</span>
            </div>
          </motion.div>
        </div>
      </section>

      {/* stat band */}
      <section className="border-y border-border">
        <div className="mx-auto grid max-w-6xl grid-cols-2 divide-x divide-border md:grid-cols-4">
          {[
            { value: cohort ? cohort.cohort.n_patients.toLocaleString() : "-", label: "synthetic patients" },
            { value: deployed ? deployed.auroc.toFixed(3) : "-", label: "AUROC, held-out patients" },
            { value: deployed ? `${Math.round(deployed.sensitivity * 100)}%` : "-", label: "events caught at 0.6 false alarms/day" },
            { value: deployed ? `${Math.round(deployed.mean_lead_min)} min` : "-", label: "mean warning time" },
          ].map((stat, index) => (
            <div key={stat.label} className={`px-6 py-8 ${index > 1 ? "border-t border-border md:border-t-0" : ""}`}>
              <div className="font-display text-[34px] leading-none">{stat.value}</div>
              <div className="mt-3 text-[11px] uppercase tracking-[0.16em] text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* problem */}
      <section id="problem" className="mx-auto max-w-6xl px-6 py-20 md:py-28">
        <div className="grid gap-12 md:grid-cols-[0.42fr_0.58fr]">
          <motion.div {...fadeUp}>
            <span className="label">01 - The problem</span>
            <h2 className="mt-4 font-display text-[36px] leading-[1.08] md:text-[42px]">
              Most of the damage happens before the reading alarms anyone.
            </h2>
          </motion.div>
          <motion.div {...fadeUp} className="space-y-6 text-[15px] leading-7 text-muted-foreground">
            <p>
              India has 101 million people with diabetes and another 136 million with prediabetes
              (ICMR-INDIAB, 2023). The patients who arrive with very high glucose usually had
              warning signs in their data days earlier: a pattern that climbs after dinner, a short
              night, a week without walks.
            </p>
            <p>
              A quarterly HbA1c cannot show any of that. A CGM stream can. But a stream on its own is
              still just a line on a screen, and a clinician with forty patients cannot watch forty lines.
            </p>
            <div className="grid gap-0 border-t border-border pt-6 sm:grid-cols-3">
              {[
                ["Static record", "Who the patient is: labs, medications, genotype, years of disease."],
                ["Sensor stream", "What the day is doing: glucose, heart rate, sleep, steps."],
                ["The twin", "Both, in one stateful model that answers 'what next?'"],
              ].map(([title, body]) => (
                <div key={title} className="border-border py-4 sm:border-r sm:px-4 sm:first:pl-0 sm:last:border-r-0">
                  <div className="text-[13px] text-foreground">{title}</div>
                  <p className="mt-1.5 text-[12px] leading-5">{body}</p>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* method */}
      <section id="method" className="border-t border-border bg-[#f7f3ea]/60">
        <div className="mx-auto max-w-6xl px-6 py-20 md:py-28">
          <motion.div {...fadeUp} className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <span className="label">02 - How it works</span>
              <h2 className="mt-4 max-w-2xl font-display text-[36px] leading-[1.08] md:text-[42px]">
                Four stages, each one a script you can run.
              </h2>
            </div>
            <p className="max-w-xs text-[12px] leading-5 text-muted-foreground">
              Python 3.10, LightGBM, a numpy fusion network, FastAPI and this React dashboard. One
              CPU, fixed seeds.
            </p>
          </motion.div>

          <div className="mt-14 grid gap-px border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
            {[
              {
                step: "01",
                title: "Generate",
                body: "1,500 Indian T2D records - earlier onset, lower BMI, higher central adiposity - each driving 14 days of 5-minute CGM, HR, HRV, steps and sleep through a meal-absorption and insulin-sensitivity model.",
              },
              {
                step: "02",
                title: "Learn",
                body: "Window features from CGM, wearables and the record feed LightGBM and a two-branch TCN + MLP network. Split by patient. Calibrated on validation, scored once on held-out patients.",
              },
              {
                step: "03",
                title: "Twin",
                body: "Each patient becomes a stateful twin: fitted insulin sensitivity and basal glucose, the current 2-hour risk, and the projected trajectory with an uncertainty band.",
              },
              {
                step: "04",
                title: "Intervene",
                body: "Why-this-alert in plain language, five what-if scenarios simulated with the patient's own parameters, and an alert timeline that shows exactly how early each warning came.",
              },
            ].map((item) => (
              <motion.div key={item.step} {...fadeUp} className="bg-background px-6 py-8">
                <div className="num text-[11px] text-muted-foreground">{item.step}</div>
                <h3 className="mt-6 font-display text-[24px]">{item.title}</h3>
                <p className="mt-3 text-[13px] leading-6 text-muted-foreground">{item.body}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* results */}
      <section id="results" className="mx-auto max-w-6xl px-6 py-20 md:py-28">
        <motion.div {...fadeUp} className="grid gap-10 md:grid-cols-[0.4fr_0.6fr]">
          <div>
            <span className="label">03 - Results</span>
            <h2 className="mt-4 font-display text-[36px] leading-[1.08] md:text-[42px]">
              The fused model carries the day. The network did not win.
            </h2>
            <p className="mt-6 text-[14px] leading-7 text-muted-foreground">
              Everything below is measured on patients the models never saw, at a common
              30-minute evaluation cadence so the comparison is fair. The deployed model is
              reported again on the dense 15-minute timeline.
            </p>
            <p className="mt-4 text-[12px] leading-5 text-muted-foreground/80">
              The static record alone ranks patients well but cannot time an alert - its
              within-patient AUROC sits at 0.500, exactly chance.
            </p>
          </div>

          <div className="border border-border bg-card">
            <div className="grid grid-cols-[1.35fr_repeat(4,0.65fr)] gap-3 border-b border-border px-4 py-3 text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
              <span>Model</span>
              <span className="text-right">AUROC</span>
              <span className="text-right">AUPRC</span>
              <span className="text-right">Sens.</span>
              <span className="text-right">Lead</span>
            </div>
            {rows.length === 0 ? (
              <div className="px-4 py-8 text-[12px] text-muted-foreground">Loading measured results...</div>
            ) : (
              rows.map((row) => (
                <div
                  key={row.model}
                  className={`grid grid-cols-[1.35fr_repeat(4,0.65fr)] items-baseline gap-3 border-b border-border/70 px-4 py-3 text-[12px] last:border-b-0 ${
                    row.model === "lgbm_fused" ? "bg-[#f7ece5]/60" : ""
                  }`}
                >
                  <span className={row.model === "lgbm_fused" ? "text-foreground" : "text-muted-foreground"}>
                    {MODEL_LABELS[row.model] ?? row.model}
                  </span>
                  <span className="num text-right">{row.auroc.toFixed(3)}</span>
                  <span className="num text-right">{row.auprc.toFixed(3)}</span>
                  <span className="num text-right">{row.sensitivity_at_far.toFixed(2)}</span>
                  <span className="num text-right">{row.mean_lead_min ? `${Math.round(row.mean_lead_min)}m` : "-"}</span>
                </div>
              ))
            )}
            <div className="grid gap-4 px-4 py-4 text-[11px] leading-5 text-muted-foreground sm:grid-cols-3">
              <div>
                <div className="label">Forecast RMSE</div>
                <div className="num mt-1 text-foreground">
                  {metrics ? `${metrics.forecast["lgbm_30min"].rmse.toFixed(1)} / ${metrics.forecast["lgbm_60min"].rmse.toFixed(1)} / ${metrics.forecast["lgbm_120min"].rmse.toFixed(1)}` : "-"}
                </div>
                <div className="mt-0.5">mg/dL at 30 / 60 / 120 min</div>
              </div>
              <div>
                <div className="label">Twin parameter fit</div>
                <div className="num mt-1 text-foreground">
                  {metrics?.param_recovery ? `${metrics.param_recovery.s_i_median_abs_pct_error.toFixed(1)}%` : "-"}
                </div>
                <div className="mt-0.5">median error, insulin sensitivity</div>
              </div>
              <div>
                <div className="label">Held-out patients</div>
                <div className="num mt-1 text-foreground">{metrics ? metrics.test_patients : "-"}</div>
                <div className="mt-0.5">14 days each, never seen in training</div>
              </div>
            </div>
          </div>
        </motion.div>

        <motion.p {...fadeUp} className="mt-10 max-w-3xl border-t border-border pt-4 text-[11px] leading-5 text-muted-foreground">
          The numbers are high because the synthetic physiology is smoother than real CGM. On real
          cohorts the same pipeline should be expected to land lower. The repository keeps the
          honest comparison: the fusion network reached AUROC 0.950 and lost, so the tree model ships.
        </motion.p>
      </section>

      {/* clinic */}
      <section id="clinic" className="border-y border-border bg-[#f7f3ea]/60">
        <div className="mx-auto grid max-w-6xl gap-12 px-6 py-20 md:grid-cols-[0.42fr_0.58fr] md:py-24">
          <motion.div {...fadeUp}>
            <span className="label">04 - In the clinic</span>
            <h2 className="mt-4 font-display text-[36px] leading-[1.08] md:text-[42px]">
              Built to be read at 9 pm, not at a conference.
            </h2>
            <p className="mt-6 text-[14px] leading-7 text-muted-foreground">
              A panel sorted by two-hour risk, a twin workspace with the trace and the projection, and
              two questions answered before the clinician asks them: why is this alert firing, and what
              changes if we act now.
            </p>
            <Link
              to={isAuthenticated ? "/dashboard" : "/auth?returnTo=%2Fdashboard"}
              className="group mt-8 inline-flex items-center gap-3 border-b border-foreground/30 pb-1 text-[12px] uppercase tracking-[0.18em] transition hover:border-foreground"
            >
              Open the dashboard
              <ArrowRight className="size-3 transition group-hover:translate-x-0.5" />
            </Link>
          </motion.div>
          <motion.div {...fadeUp} className="grid gap-px bg-border sm:grid-cols-2">
            {[
              ["Panel", "Sixty patients, ranked by the next two hours. Status chips, trend, time in range, alerts in the last 72 hours."],
              ["Twin", "CGM trace with the projected two hours and its band, a risk gauge, sleep, HRV and steps."],
              ["Why this alert", "The five features pushing the risk up or down, in plain language, from SHAP values."],
              ["What if", "Walk after dinner, swap rice for millets, skip the snack, sleep seven hours, adjust metformin. Simulated, with the event probability before and after."],
            ].map(([title, body]) => (
              <div key={title} className="bg-background px-6 py-7">
                <div className="text-[13px]">{title}</div>
                <p className="mt-2 text-[12px] leading-6 text-muted-foreground">{body}</p>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* guardrails */}
      <section className="mx-auto max-w-6xl px-6 py-20">
        <motion.div {...fadeUp} className="grid gap-10 md:grid-cols-[0.42fr_0.58fr]">
          <div>
            <span className="label">05 - Guardrails</span>
            <h2 className="mt-4 font-display text-[32px] leading-[1.1]">
              What this is, and what it is not.
            </h2>
          </div>
          <div className="grid gap-6 sm:grid-cols-2">
            {[
              ["Synthetic only", "All 1,500 patients are generated by this repository. No real records, no scraped data."],
              ["Privacy by design", "Built to DPDP Act and HIPAA expectations: no identifiers, no external calls, data stays local."],
              ["Not clinical advice", "Every scenario is a simulation for decision support. A clinician decides."],
              ["Not validated", "Synthetic data, no prospective trial. The next step is an open CGM dataset and then a supervised pilot."],
            ].map(([title, body]) => (
              <div key={title} className="border-t border-border pt-4">
                <div className="text-[13px]">{title}</div>
                <p className="mt-2 text-[12px] leading-6 text-muted-foreground">{body}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* closing */}
      <section className="border-t border-border">
        <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-8 px-6 py-16 md:flex-row md:items-center">
          <div>
            <h2 className="font-display text-[34px] leading-tight md:text-[40px]">Look at the panel.</h2>
            <p className="mt-2 text-[13px] text-muted-foreground">
              {cohort
                ? `${cohort.patients.length} synthetic patients in the demo panel, ${
                    cohort.patients.filter((row) => row.status === "critical").length
                  } of them currently flagged critical.`
                : "Synthetic patients, ranked by the next two hours."}
            </p>
          </div>
          <InkButton to={isAuthenticated ? "/dashboard" : "/auth?returnTo=%2Fdashboard"}>
            Enter the clinic
          </InkButton>
        </div>
      </section>

      <footer className="border-t border-border">
        <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-8 text-[11px] text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <span className="font-display text-[15px] text-foreground">GlucoTwin</span>
            <span>MIT licence</span>
            <span>Synthetic data, generated by this repository</span>
          </div>
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <span>six seven - IGDTUW</span>
            <span>Summit 2026 proof of concept</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
