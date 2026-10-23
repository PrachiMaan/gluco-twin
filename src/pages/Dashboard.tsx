import { useEffect, useState } from "react";
import { Link } from "react-router";
import { LogOut, PanelLeftClose, ShieldAlert, Users } from "lucide-react";

import { CohortView } from "@/components/clinic/CohortView";
import { TwinWorkspace } from "@/components/clinic/TwinWorkspace";
import { useAuth } from "@/hooks/use-auth";
import {
  loadCohort,
  loadPatient,
  type CohortPayload,
  type PatientBundle,
} from "@/lib/clinic";

function readPatientParam(): string | null {
  if (typeof window === "undefined") return null;
  return new URLSearchParams(window.location.search).get("patient");
}

function writePatientParam(id: string | null) {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  if (id) url.searchParams.set("patient", id);
  else url.searchParams.delete("patient");
  window.history.replaceState({}, "", url.toString());
}

function LoadingLine({ label }: { label: string }) {
  return (
    <div className="flex h-[60vh] flex-col items-center justify-center gap-3">
      <div className="h-px w-10 animate-pulse bg-foreground/40" />
      <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
    </div>
  );
}

export default function Dashboard() {
  const { user, signOut } = useAuth();
  const [cohort, setCohort] = useState<CohortPayload | null>(null);
  const [patient, setPatient] = useState<PatientBundle | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [minute, setMinute] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loadingPatient, setLoadingPatient] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    loadCohort()
      .then((payload) => {
        if (!alive) return;
        setCohort(payload);
        const requested = readPatientParam();
        const exists = requested && payload.patients.some((row) => row.id === requested);
        const initial = exists ? requested : payload.patients[0]?.id ?? null;
        if (initial) setSelectedId(initial);
      })
      .catch((cause: Error) => alive && setError(cause.message));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setPatient(null);
      setPlaying(false);
      return;
    }
    let alive = true;
    setLoadingPatient(true);
    writePatientParam(selectedId);
    loadPatient(selectedId)
      .then((bundle) => {
        if (!alive) return;
        setPatient(bundle);
        setMinute(bundle.summary.now_minute);
        setPlaying(false);
        setLoadingPatient(false);
      })
      .catch((cause: Error) => {
        if (!alive) return;
        setError(cause.message);
        setLoadingPatient(false);
      });
    return () => {
      alive = false;
    };
  }, [selectedId]);

  useEffect(() => {
    if (!playing || !patient) return;
    if (minute >= patient.summary.now_minute) {
      setPlaying(false);
      return;
    }
    const handle = window.setTimeout(() => {
      setMinute((current) => Math.min(current + 5, patient.summary.now_minute));
    }, 420);
    return () => window.clearTimeout(handle);
  }, [playing, patient, minute]);

  const counts = cohort?.patients.reduce(
    (acc, row) => {
      acc[row.status] += 1;
      return acc;
    },
    { critical: 0, watch: 0, stable: 0 } as Record<"critical" | "watch" | "stable", number>,
  );

  const selectPatient = (id: string) => {
    setSelectedId(id);
    setPatient(null);
  };

  const handlePlaying = (next: boolean) => {
    if (!patient) return;
    if (next && minute >= patient.summary.now_minute) setMinute(0);
    setPlaying(next);
  };

  return (
    <div className="min-h-screen bg-[#f6f2e9] text-foreground">
      <div className="mx-auto flex min-h-screen max-w-[1600px]">
        {/* sidebar */}
        <aside className="sticky top-0 hidden h-screen w-[236px] shrink-0 flex-col justify-between border-r border-border bg-card/60 px-5 py-6 lg:flex">
          <div>
            <Link to="/" className="flex items-baseline gap-1.5">
              <span className="font-display text-[22px] leading-none">GlucoTwin</span>
              <span className="size-1 rounded-full bg-[#b4552d]" aria-hidden />
            </Link>
            <p className="mt-1.5 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">Clinic workspace</p>

            <nav className="mt-9 space-y-1">
              <button
                type="button"
                onClick={() => {
                  setSelectedId(null);
                  setPatient(null);
                  writePatientParam(null);
                }}
                className={`flex w-full items-center gap-2.5 border-l px-3 py-2 text-left text-[12px] transition-colors ${
                  selectedId
                    ? "border-transparent text-muted-foreground hover:text-foreground"
                    : "border-foreground/70 bg-accent/50 text-foreground"
                }`}
              >
                <Users className="size-3.5" />
                Cohort
              </button>
              <div
                className={`flex items-center gap-2.5 border-l px-3 py-2 text-[12px] ${
                  selectedId ? "border-foreground/70 bg-accent/50 text-foreground" : "border-transparent text-muted-foreground/60"
                }`}
              >
                <PanelLeftClose className="size-3.5" />
                Twin workspace
              </div>
            </nav>

            <div className="mt-8 border-t border-border pt-4">
              <p className="text-[9px] uppercase tracking-[0.16em] text-muted-foreground">Flagged now</p>
              <ul className="mt-3 space-y-2 text-[11px]">
                {[
                  ["critical", "Critical", "#b4552d"],
                  ["watch", "Watch", "#b08a4f"],
                  ["stable", "Stable", "#6f8060"],
                ].map(([key, label, color]) => (
                  <li key={key} className="flex items-center justify-between">
                    <span className="flex items-center gap-2 text-muted-foreground">
                      <span className="size-1.5 rounded-full" style={{ backgroundColor: color }} />
                      {label}
                    </span>
                    <span className="num">{counts ? counts[key as "critical" | "watch" | "stable"] : "-"}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="mt-8 border border-border/80 p-3">
              <p className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                <ShieldAlert className="size-3" />
                Synthetic data
              </p>
              <p className="mt-2 text-[10px] leading-4 text-muted-foreground">
                1,500 generated patients. No real records. Research prototype, not a medical device.
              </p>
            </div>
          </div>

          <div className="border-t border-border pt-4">
            <p className="truncate text-[11px] text-muted-foreground">{user?.email ?? "Signed in"}</p>
            <div className="mt-2 flex items-center justify-between">
              <Link to="/" className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground transition-colors hover:text-foreground">
                Project page
              </Link>
              <button
                type="button"
                onClick={() => void signOut()}
                className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-[0.12em] text-muted-foreground transition-colors hover:text-foreground"
              >
                <LogOut className="size-3" />
                Sign out
              </button>
            </div>
          </div>
        </aside>

        {/* main */}
        <div className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-20 flex h-14 items-center justify-between gap-4 border-b border-border bg-[#f6f2e9]/92 px-4 backdrop-blur sm:px-6">
            <div className="flex items-center gap-3">
              <Link to="/" className="font-display text-[18px] leading-none lg:hidden">
                GlucoTwin
              </Link>
              <span className="hidden text-[10px] uppercase tracking-[0.16em] text-muted-foreground lg:inline">
                {selectedId ? `Panel / ${selectedId}` : "Panel / all patients"}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="hidden border border-border px-2 py-1 text-[9px] uppercase tracking-[0.14em] text-muted-foreground sm:inline">
                Research prototype
              </span>
              <span className="num text-[11px] text-muted-foreground">
                {cohort ? `${cohort.patients.length} patients` : ""}
              </span>
              <button
                type="button"
                onClick={() => void signOut()}
                className="inline-flex size-7 items-center justify-center border border-border text-muted-foreground transition-colors hover:border-foreground/40 hover:text-foreground lg:hidden"
                title="Sign out"
              >
                <LogOut className="size-3.5" />
              </button>
            </div>
          </header>

          <main className="flex-1 px-4 py-6 sm:px-6 sm:py-8">
            {error ? (
              <div className="border border-[#d9b6a3] bg-[#f7ece5] px-4 py-3 text-[12px] text-[#8f3d1c]">
                {error}
              </div>
            ) : !cohort ? (
              <LoadingLine label="Loading the synthetic panel" />
            ) : selectedId && (!patient || patient.id !== selectedId) ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h1 className="num text-[24px]">{selectedId}</h1>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedId(null);
                      writePatientParam(null);
                    }}
                    className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground hover:text-foreground"
                  >
                    Back to cohort
                  </button>
                </div>
                <div className="border border-border bg-card px-4 py-16 text-center text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
                  {loadingPatient ? "Reconstructing the twin" : "No twin loaded"}
                </div>
              </div>
            ) : patient ? (
              <TwinWorkspace
                patient={patient}
                minute={minute}
                playing={playing}
                onMinute={setMinute}
                onPlaying={handlePlaying}
                onBack={() => {
                  setSelectedId(null);
                  setPatient(null);
                  setPlaying(false);
                  writePatientParam(null);
                }}
              />
            ) : (
              <CohortView cohort={cohort} selectedId={selectedId} onSelect={selectPatient} />
            )}
          </main>

          <footer className="border-t border-border px-4 py-3 text-[10px] text-muted-foreground sm:px-6">
            GlucoTwin · synthetic proof of concept · DPDP Act and HIPAA by design · a clinician makes every decision.
          </footer>
        </div>
      </div>
    </div>
  );
}
