import type { Metadata } from "next";
import localFont from "next/font/local";
import SafeClerkProvider from "@/components/auth/SafeClerkProvider";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Rootix — AI Distributed Trace Debugger",
  description:
    "AI-powered root cause analysis for distributed traces. " +
    "Identifies N+1 queries, slow queries, retry storms, and cascading failures.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || "";

  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased`}
      >
        <SafeClerkProvider publishableKey={publishableKey || undefined}>
          {children}
        </SafeClerkProvider>
      </body>
    </html>
  );
}
