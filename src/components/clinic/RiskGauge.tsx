import type { Status } from "@/lib/clinic";

const TONES: Record<Status, string> = {
  critical: "#b4552d",
  watch: "#b08a4f",
  stable: "#6f8060",
};

export function RiskGauge({
  risk,
  status,
  label = "2-hour event risk",
}: {
  risk: number;
  status: Status;
  label?: string;
}) {
  const radius = 74;
  const cx = 92;
  const cy = 92;
  const length = Math.PI * radius;
  const filled = Math.max(0.005, Math.min(1, risk)) * length;
  const color = TONES[status];

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 184 104" className="w-full max-w-[220px]" role="img" aria-label={`${label}: ${Math.round(risk * 100)} percent`}>
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke="var(--border)"
          strokeWidth="10"
          strokeLinecap="butt"
        />
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="butt"
          strokeDasharray={`${filled} ${length}`}
          style={{ transition: "stroke-dasharray 600ms ease, stroke 400ms ease" }}
        />
        {[0, 0.35, 0.7].map((mark) => {
          const angle = Math.PI * (1 - mark);
          const x1 = cx + (radius - 12) * Math.cos(angle);
          const y1 = cy - (radius - 12) * Math.sin(angle);
          const x2 = cx + (radius + 2) * Math.cos(angle);
          const y2 = cy - (radius + 2) * Math.sin(angle);
          return <line key={mark} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--border)" strokeWidth="1" />;
        })}
      </svg>
      <div className="-mt-6 text-center">
        <div className="num text-[30px] leading-none" style={{ color }}>
          {Math.round(risk * 100)}
          <span className="ml-0.5 text-[14px] text-muted-foreground">%</span>
        </div>
        <div className="mt-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{label}</div>
      </div>
    </div>
  );
}
