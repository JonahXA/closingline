"use client";

import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { tooltipStyle, usePalette } from "./palette";

type Bin = { bin_mid: number; predicted: number; observed: number; n: number };

export default function Calibration({ bins }: { bins: Bin[] }) {
  const p = usePalette();
  const data = bins.map((b) => ({ ...b, ideal: b.predicted }));
  const pct = (v: number) => `${Math.round(v * 100)}%`;

  return (
    <div className="chart-scroll">
      <ResponsiveContainer width="100%" height={300} minWidth={420}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 14, left: 0 }}>
          <CartesianGrid stroke={p.grid} strokeWidth={1} vertical={false} />
          <XAxis
            dataKey="predicted"
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={pct}
            tick={{ fill: p.muted, fontSize: 12 }}
            stroke={p.baseline}
            label={{ value: "Forecast probability", position: "insideBottom", dy: 10, fill: p.muted, fontSize: 12 }}
            height={44}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            tickFormatter={pct}
            tick={{ fill: p.muted, fontSize: 12 }}
            stroke={p.baseline}
            label={{ value: "Observed frequency", angle: -90, position: "insideLeft", dx: 8, fill: p.muted, fontSize: 12 }}
          />
          <Tooltip
            contentStyle={tooltipStyle(p)}
            formatter={(v: number, name: string) =>
              name === "ideal" ? [] : [pct(v), name === "observed" ? "Observed" : name]
            }
            labelFormatter={(l: number) => `Forecast ${pct(l)}`}
          />
          <Line
            dataKey="ideal"
            stroke={p.baseline}
            strokeDasharray="5 4"
            strokeWidth={1.5}
            dot={false}
            activeDot={false}
            isAnimationActive={false}
            name="ideal"
          />
          <Line
            dataKey="observed"
            stroke={p.model}
            strokeWidth={2}
            dot={{ r: 4, fill: p.model, stroke: p.surface, strokeWidth: 2 }}
            isAnimationActive={false}
            name="observed"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
