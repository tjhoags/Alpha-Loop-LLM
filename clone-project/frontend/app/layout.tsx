import "./globals.css";
import { ReactNode } from "react";

export const metadata = {
  title: "InstantMeet",
  description: "Payments-first scheduling with team routing.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

