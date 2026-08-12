import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the TSQA inspection shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html lang="zh-CN">/);
  assert.match(html, /<title>TSQA Lens · 时间序列QA数据审查台<\/title>/);
  assert.match(html, /TSQA Lens/);
  assert.match(html, /OpenTSLM/);
  assert.match(html, /同模板成员/);
  assert.match(html, /中译问题与答案/);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
});

test("source wires record, template and Qwen translation APIs", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const config = await readFile(new URL("../inspector_config.yaml", import.meta.url), "utf8");

  assert.match(page, /\/api\/record\?dataset=/);
  assert.match(page, /\/api\/template-members\?dataset=/);
  assert.match(page, /\/api\/translate/);
  assert.match(page, /choice_\$\{label\}/);
  assert.match(page, /<SeriesChart/);
  assert.match(config, /Qwen3\.6-27B/);
  assert.match(config, /frontend_origin:\s*"\*"/);
});
