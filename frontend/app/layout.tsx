import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "AgriTech AI — Smart Crop Recommendation System",
  description:
    "AI-powered crop recommendation system using Machine Learning. Enter soil nutrients and weather conditions to get instant crop predictions. Built with Random Forest, Flask, and Next.js.",
  keywords: [
    "crop recommendation",
    "machine learning",
    "agriculture AI",
    "random forest",
    "agritech",
    "smart farming",
    "crop prediction",
  ],
  authors: [{ name: "Yasir", url: "https://github.com/sirrryasir" }],
  openGraph: {
    title: "AgriTech AI — Smart Crop Recommendation",
    description:
      "ML-powered crop recommendation based on soil & weather data. 99%+ accuracy with Random Forest.",
    type: "website",
    url: "https://agritech-ai-sir.vercel.app",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
