"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { tooltipStyle, usePalette } from "./palette";

type Row = {
  league: string;
  matches: number;
  model_brier: number;
  market_brier: number;
};

export default function BrierBars({
  rows,
  names,
}: {
  rows: Row[];
  names: Record<string, string>;
}) {
  const p = usePalette();
  const data = rows
    .filter((r) => r.league !== "ALL")
    .map((r) => ({ ...r, name: names[r.league] ?? r.league }));

  return (
    <>
      <div className="legend">
        <span>
          <span className="swatch" style={{ background: "var(--series-model)" }} />
          ClosingLine model
        </span>
        <span>
          <span className="swatch" style={{ background: "var(--series-market)" }} />
          Closing line (de-vigged)
        </span>
      </div>
      <div className="chart-scroll">
        <ResponsiveContainer width="100%" height={260} minWidth={420}>
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barGap={2}>
            <CartesianGrid stroke={p.grid} strokeWidth={1} vertical={false} />
            <XAxis dataKey="name" tick={{ fill: p.muted, fontSize: 12 }} stroke={p.baseline} />
            <YAxis
              domain={[0, 0.72]}
              tick={{ fill: p.muted, fontSize: 12 }}
              stroke={p.baseline}
              width={44}
            />
            <Tooltip
              cursor={{ fill: p.grid, opacity: 0.4 }}
              contentStyle={tooltipStyle(p)}
              formatter={(v: number, name: string) => [
                v.toFixed(4),
                name === "model_brier" ? "Model Brier" : "Market Brier",
              ]}
            />
            <Bar dataKey="model_brier" fill={p.model} radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false} />
            <Bar dataKey="market_brier" fill={p.market} radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
