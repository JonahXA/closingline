"use client";

import { useEffect, useState } from "react";

// Chart colors resolved in JS because SVG presentation attributes can't
// consume CSS custom properties. Values mirror globals.css.
const LIGHT = {
  model: "#2a78d6",
  market: "#1baf7a",
  ink: "#0b0b0b",
  ink2: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  baseline: "#c3c2b7",
  surface: "#fcfcfb",
};

const DARK: typeof LIGHT = {
  model: "#3987e5",
  market: "#199e70",
  ink: "#ffffff",
  ink2: "#c3c2b7",
  muted: "#898781",
  grid: "#2c2c2a",
  baseline: "#383835",
  surface: "#1a1a19",
};

export function usePalette() {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    setDark(mq.matches);
    const fn = (e: MediaQueryListEvent) => setDark(e.matches);
    mq.addEventListener("change", fn);
    return () => mq.removeEventListener("change", fn);
  }, []);
  return dark ? DARK : LIGHT;
}

export const tooltipStyle = (p: ReturnType<typeof usePalette>) => ({
  background: p.surface,
  border: `1px solid ${p.baseline}`,
  borderRadius: 8,
  fontSize: 12,
  color: p.ink,
});
