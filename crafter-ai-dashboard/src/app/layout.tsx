import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";

export const metadata: Metadata = {
  title: "CRAfter AI — Visual Reinforcement Learning Lab",
  description: "Interactive Reinforcement Learning Visual Lab for Crafter",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-neutral-100 flex min-h-screen antialiased">
        <Sidebar />
        <main className="flex-1 min-w-0 flex flex-col h-screen overflow-y-auto">
          {children}
        </main>
      </body>
    </html>
  );
}

