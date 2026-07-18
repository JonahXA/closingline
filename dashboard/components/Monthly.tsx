"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { tooltipStyle, usePalette } from "./palette";

type Row = { month: string; model_brier: number; market_brier: number; n: number };

export default function Monthly({ rows }: { rows: Row[] }) {
  const p = usePalette();

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
        <ResponsiveContainer width="100%" height={280} minWidth={520}>
          <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={p.grid} strokeWidth={1} vertical={false} />
            <XAxis
              dataKey="month"
              tick={{ fill: p.muted, fontSize: 11 }}
              stroke={p.baseline}
              interval="preserveStartEnd"
              minTickGap={28}
            />
            <YAxis
              domain={["auto", "auto"]}
              tick={{ fill: p.muted, fontSize: 12 }}
              stroke={p.baseline}
              width={48}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <Tooltip
              contentStyle={tooltipStyle(p)}
              formatter={(v: number, name: string) => [
                v.toFixed(4),
                name === "model_brier" ? "Model Brier" : "Market Brier",
              ]}
              labelFormatter={(l: string, payload) =>
                `${l} · ${payload?.[0]?.payload?.n ?? "?"} matches`
              }
            />
            <Line
              dataKey="model_brier"
              stroke={p.model}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, stroke: p.surface, strokeWidth: 2 }}
              isAnimationActive={false}
            />
            <Line
              dataKey="market_brier"
              stroke={p.market}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, stroke: p.surface, strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
