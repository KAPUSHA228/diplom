import { useState, useEffect } from "react";
import * as XLSX from "xlsx";
import { selectSubset } from "../api";
import { parseFile } from "../utils/parseFile";
import { useSharedData } from "../hooks/useSharedData";

async function parseFileFull(file) {
  const isExcel = file.name.endsWith(".xlsx") || file.name.endsWith(".xls");
  if (isExcel) {
    const buf = await file.arrayBuffer();
    const wb = XLSX.read(buf, { type: "array" });
    return XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { defval: null });
  }
  const text = await file.text();
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map(h => h.trim());
  return lines.slice(1).map(line => {
    const cells = line.split(",").map(c => c.trim());
    const obj = {};
    headers.forEach((h, i) => { const v = cells[i] ?? ""; obj[h] = v === "" ? null : (isNaN(v) ? v : Number(v)); });
    return obj;
  });
}

const MODES = [
  { value: "random", label: "🎲 Случайная выборка" },
  { value: "query", label: "🔍 По условию" },
  { value: "cluster", label: "🏷️ По кластеру" },
];

export default function SubsetSelect() {
  const { data: sharedData, hasShared } = useSharedData();
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [mode, setMode] = useState("random");
  const [nSamples, setNSamples] = useState(50);
  const [query, setQuery] = useState("");
  const [clusterId, setClusterId] = useState(0);
  const [randomSeed, setRandomSeed] = useState(42); // По умолчанию фиксированный
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [clusterCount, setClusterCount] = useState(0);

  // Читаем количество кластеров из последнего результата анализа
  useEffect(() => {
    try {
      const raw = localStorage.getItem("last_analysis_result");
      const lastAnalysis = raw ? JSON.parse(raw) : null;
      if (lastAnalysis?.cluster_profiles?.size) {
        // cluster_profiles = { "Признак1": { "0": ..., "1": ... }, "size": { "0": ..., "1": ... } }
        // Количество кластеров = количество ключей внутри "size"
        const count = Object.keys(lastAnalysis.cluster_profiles.size).length;
        setClusterCount(count);
      }
    } catch { /* ignore */ }
  }, []);

  // Генерируем список доступных кластеров динамически
  const availableClusters = clusterCount > 0
    ? Array.from({ length: clusterCount }, (_, i) => i)
    : [];

  const activeData = fileData || sharedData;

  useEffect(() => {
    if (hasShared && !fileData) setNSamples(Math.min(50, sharedData?.length || 50));
  }, [hasShared, fileData, sharedData]);

  async function onFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setResult(null); setError("");
    try {
      const rows = await parseFileFull(f);
      setFileData(rows);
    } catch (err) { setError("Ошибка: " + err.message); }
  }

  async function onRun() {
    if (!activeData?.length) return;
    setBusy(true); setError(""); setResult(null);
    try {
      let condition = null, n = null, byCluster = null;
      if (mode === "random") n = nSamples;
      else if (mode === "query") { if (!query.trim()) { setError("Введите условие"); setBusy(false); return; } condition = query.trim(); }
      else if (mode === "cluster") byCluster = clusterId;

      const res = await selectSubset(activeData, condition, n, byCluster, randomSeed);
      setResult(res);
    } catch (err) { setError("Ошибка: " + err.message); }
    finally { setBusy(false); }
  }

  return (
    <div className="card">
      <h2>📋 Выделение подмножества</h2>

      <div style={{ marginBottom: 12 }}>
        {hasShared && !fileData && (
          <div className="ok" style={{ padding: 8, borderRadius: 6, background: "var(--bg-secondary)", marginBottom: 8 }}>
            ✅ Данные с главной: <b>{sharedData.length} строк</b>
          </div>
        )}
        <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileChange} />
      </div>

      <div style={{ marginBottom: 12 }}>
        <label><b>Режим:</b></label>
        <div className="strategy-grid" style={{ marginTop: 8 }}>
          {MODES.map(m => (
            <label key={m.value} className={`strategy-option ${mode === m.value ? "selected" : ""}`}>
              <input type="radio" name="subset-mode" value={m.value} checked={mode === m.value} onChange={() => setMode(m.value)} />{m.label}
            </label>
          ))}
        </div>
      </div>

      {mode === "random" && (
        <div style={{ marginBottom: 12 }}>
          <label><b>Кол-во:</b> {nSamples}
            <input type="range" min="5" max={Math.min(activeData?.length || 500, 500)} step="5" value={nSamples} onChange={e => setNSamples(Number(e.target.value))} style={{ width: "100%", marginTop: 4 }} />
          </label>
          <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
            <label style={{ fontSize: 13 }}>Seed:
              <input type="number" value={randomSeed} onChange={e => setRandomSeed(Number(e.target.value))}
                style={{ width: 80, marginLeft: 4, padding: 4, borderRadius: 4, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)" }} />
            </label>
            <button type="button" onClick={() => setRandomSeed(Math.floor(Math.random() * 10000))} style={{ padding: "4px 8px", borderRadius: 4, border: "1px solid var(--border)", background: "var(--bg-secondary)", color: "var(--text)", cursor: "pointer" }}>
              🎲 Рандом
            </button>
          </div>
        </div>
      )}
      {mode === "query" && (
        <div style={{ marginBottom: 12 }}>
          <label><b>Условие:</b>
            <input type="text" value={query} onChange={e => setQuery(e.target.value)} placeholder="avg_grade > 4 and risk_flag == 0" style={{ width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)" }} />
          </label>
        </div>
      )}
      {mode === "cluster" && (
        <div style={{ marginBottom: 12 }}>
          <label><b>Кластер:</b>
            {availableClusters.length > 0 ? (
              <select value={clusterId} onChange={e => setClusterId(Number(e.target.value))}
                style={{ width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)" }}>
                {availableClusters.map(c => <option key={c} value={c}>Кластер {c}</option>)}
              </select>
            ) : (
              <span className="muted" style={{ fontSize: 13 }}> (Сначала запустите анализ на главной)</span>
            )}
          </label>
        </div>
      )}

      <button className="primary" onClick={onRun} disabled={!activeData || busy || (mode === "cluster" && availableClusters.length === 0)}>
        {busy ? "⏳ ..." : "🚀 Выделить"}
      </button>
      {error && <p className="error" style={{ marginTop: 8 }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 12 }}>
          <p className="ok">✅ Найдено: <b>{result.count}</b> студентов</p>
          {result.data?.length > 0 && (
            <div className="table-wrap"><table><thead><tr>{Object.keys(result.data[0]).map(h => <th key={h}>{h}</th>)}</tr></thead>
              <tbody>{result.data.slice(0, 20).map((row, i) => (<tr key={i}>{Object.values(row).map((v, j) => <td key={j}>{v !== null && v !== undefined ? String(v) : ""}</td>)}</tr>))}</tbody></table>
              {result.count > 20 && <p className="muted">Показано 20 из {result.count}</p>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
