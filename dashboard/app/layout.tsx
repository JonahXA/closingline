import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ClosingLine — pre-registered football forecasts vs the market",
  description:
    "Live probabilistic forecasts for the Big-5 European leagues, frozen in git before kickoff and benchmarked against the sportsbook closing line.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
