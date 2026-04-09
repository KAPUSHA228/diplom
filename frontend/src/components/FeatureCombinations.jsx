import { useState, useEffect } from "react";
import * as XLSX from "xlsx";
import { createFeatureCombinations } from "../api";
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

export default function FeatureCombinations() {
  const { data: sharedData, hasShared, updateData } = useSharedData();
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [numericalCols, setNumericalCols] = useState([]);
  const [textCols, setTextCols] = useState([]);
  const [maxPairs, setMaxPairs] = useState(15);
  const [targetCol, setTargetCol] = useState("");
  const [allCols, setAllCols] = useState([]);

  const [result, setResult] = useState(null);
  const [selectedRecs, setSelectedRecs] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const activeData = fileData || sharedData;

  // Авто-загрузка целевой переменной из последнего анализа
  useEffect(() => {
    try {
      const last = JSON.parse(localStorage.getItem("last_analysis_result"));
      if (last?.target_col) setTargetCol(last.target_col);
    } catch {}
  }, []);

  useEffect(() => {
    if (activeData?.length) {
      const cols = Object.keys(activeData[0]);
      setAllCols(cols);
      setNumericalCols(cols.filter(c => typeof activeData[0][c] === "number"));
      setTextCols(cols.filter(c => typeof activeData[0][c] === "string"));
    }
  }, [activeData]);

  async function onFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setResult(null); setError("");
    try {
      const rows = await parseFileFull(f);
      setFileData(rows);
    } catch (err) { setError("Ошибка чтения файла: " + err.message); }
  }

  async function onRun() {
    if (!activeData) return;
    setBusy(true); setError(""); setResult(null); setSelectedRecs([]);
    try {
      const res = await createFeatureCombinations(activeData, numericalCols, textCols, maxPairs, targetCol || null);
      setResult(res);
      // По умолчанию выбираем все рекомендации
      if (res.recommendations) setSelectedRecs(res.recommendations.map(r => r.name));
    } catch (err) { setError("Ошибка: " + err.message); }
    finally { setBusy(false); }
  }

  async function onAddSelected() {
    if (!result || !selectedRecs.length) return;
    try {
      // Берем только выбранные колонки из полного ответа и добавляем к активным данным
      const newData = activeData.map((row, idx) => {
        const newRow = { ...row };
        selectedRecs.forEach(col => {
          if (result.data[idx]) newRow[col] = result.data[idx][col];
        });
        return newRow;
      });
      updateData(newData);
      if (fileData) setFileData(newData);
      alert(`✅ Добавлено ${selectedRecs.length} новых признаков!`);
    } catch (e) { setError("Ошибка сохранения: " + e.message); }
  }

  return (
    <div className="card">
      <h2>🔗 Комбинации признаков (Smart)</h2>
      <p className="muted">Система найдет комбинации, которые сильнее всего влияют на цель.</p>

      <div style={{ marginBottom: 12 }}>
        {hasShared && !fileData && <div className="ok" style={{ padding: 8, borderRadius: 6, background: "var(--bg-secondary)", marginBottom: 8 }}>✅ Данные с главной: <b>{sharedData.length} строк</b></div>}
        <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileChange} />
      </div>

      {allCols.length > 0 && (
        <>
          <div style={{ marginBottom: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <label><b>Целевая переменная:</b>
              <select value={targetCol} onChange={e => setTargetCol(e.target.value)}
                style={{ width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)" }}>
                <option value="">— Не выбрана (генерировать всё) —</option>
                {allCols.filter(c => typeof activeData[0][c] === "number").map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label><b>Макс. комбинаций:</b> {maxPairs}
              <input type="range" min="5" max="50" step="1" value={maxPairs} onChange={e => setMaxPairs(Number(e.target.value))} style={{ width: "100%", marginTop: 4 }} />
            </label>
          </div>
          <button className="primary" onClick={onRun} disabled={!activeData || busy}>{busy ? "⏳ Анализ..." : "🔍 Найти лучшие связи"}</button>
        </>
      )}

      {error && <p className="error" style={{ marginTop: 8 }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 16, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
          <h3>🏆 Рекомендации системы</h3>
          {result.recommendations?.length > 0 ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 16 }}>
              {result.recommendations.map(rec => (
                <label key={rec.name} style={{ display: "flex", alignItems: "center", gap: 10, padding: 8, borderRadius: 6, background: "var(--bg-secondary)" }}>
                  <input type="checkbox" checked={selectedRecs.includes(rec.name)}
                    onChange={e => setSelectedRecs(prev => e.target.checked ? [...prev, rec.name] : prev.filter(x => x !== rec.name))} />
                  <span style={{ fontWeight: 600 }}>{rec.name}</span>
                  <span style={{ marginLeft: "auto", color: rec.correlation > 0 ? "#2ecc71" : "#e74c3c", fontWeight: "bold" }}>
                    r = {rec.correlation > 0 ? "+" : ""}{rec.correlation}
                  </span>
                </label>
              ))}
            </div>
          ) : <p className="muted">Сильных связей с целью не найдено.</p>}

          <button onClick={onAddSelected} disabled={!selectedRecs.length} style={{ padding: "8px 16px", borderRadius: 6, border: "none", background: selectedRecs.length ? "var(--primary)" : "#666", color: "#fff", cursor: "pointer" }}>
            💾 Добавить выбранное в набор данных ({selectedRecs.length})
          </button>
        </div>
      )}
    </div>
  );
}
