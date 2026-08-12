import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TSQA Lens · 时间序列QA数据审查台",
  description: "逐条审查时间序列QA、模板重复、标签质量与Qwen中文翻译。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
