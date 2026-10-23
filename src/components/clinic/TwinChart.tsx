import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatClock, type SeriesPoint } from "@/lib/clinic";

interface TwinChartProps {
  points: SeriesPoint[];
  nowLabel?: number;
  height?: number;
  showBand?: boolean;
  xKind?: "clock" | "day";
}

export function TwinChart({ points, nowLabel, height = 320, showBand = true, xKind = "clock" }: TwinChartProps) {
  // recharts renders a range area when the dataKey holds a [low, high] tuple
  const data = points.map((point) => ({
    ...point,
    band: point.lo !== null && point.hi !== null ? [point.lo, point.hi] : null,
  }));
  const formatX = (value: number) =>
    xKind === "day" ? `D${Math.floor(value / 1440) + 1}` : formatClock(value);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 10, right: 14, bottom: 2, left: -6 }}>
        <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="2 5" />
        <XAxis
          dataKey="label"
          tickFormatter={(value: number) => formatX(value)}
          minTickGap={54}
          axisLine={false}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={[40, 320]}
          ticks={[70, 120, 180, 240, 300]}
          width={38}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ stroke: "var(--foreground)", strokeOpacity: 0.15, strokeDasharray: "2 3" }}
          contentStyle={{
            border: "1px solid var(--border)",
            borderRadius: 2,
            background: "var(--card)",
            fontSize: 11,
            boxShadow: "none",
          }}
          labelFormatter={(value) => formatX(Number(value))}
          formatter={(value, name) => {
            if (name === "band") {
              if (Array.isArray(value)) {
                return [`${Math.round(Number(value[0]))} - ${Math.round(Number(value[1]))} mg/dL`, "projected"];
              }
              return [null, null];
            }
            if (name === "glucose" && value !== null && value !== undefined) {
              return [`${Math.round(Number(value))} mg/dL`, "glucose"];
            }
            return [null, null];
          }}
        />
        <ReferenceLine
          y={180}
          stroke="var(--chart-1)"
          strokeDasharray="2 4"
          strokeOpacity={0.45}
          label={{ value: "180", position: "insideTopLeft", fill: "var(--chart-1)", fontSize: 9 }}
        />
        <ReferenceLine y={70} stroke="var(--chart-3)" strokeDasharray="2 4" strokeOpacity={0.4} />
        {showBand ? (
          <Area
            type="monotone"
            dataKey="band"
            stroke="none"
            fill="var(--chart-3)"
            fillOpacity={0.12}
            isAnimationActive={false}
            connectNulls={false}
          />
        ) : null}
        <Line
          type="monotone"
          dataKey="glucose"
          stroke="var(--chart-1)"
          strokeWidth={1.7}
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
        {nowLabel !== undefined ? (
          <ReferenceLine
            x={nowLabel}
            stroke="var(--foreground)"
            strokeOpacity={0.3}
            strokeDasharray="1 3"
            label={{ value: "now", position: "insideTopRight", fill: "var(--muted-foreground)", fontSize: 9 }}
          />
        ) : null}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
