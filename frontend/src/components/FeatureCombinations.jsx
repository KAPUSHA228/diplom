import { useState, useEffect } from "react";
import * as XLSX from "xlsx";
import { createFeatureCombinations } from "../api";
import { parseFile } from "../utils/parseFile";
import { useSharedData } from "../hooks/useSharedData";

export default function FeatureCombinations() {
  const { data: sharedData, columns: sharedCols, hasShared } = useSharedData();
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [numericalCols, setNumericalCols] = useState([]);
  const [textCols, setTextCols] = useState([]);
  const [maxPairs, setMaxPairs] = useState(15);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const activeData = fileData || sharedData;
  const columns = fileData ? Object.keys(fileData[0] || {}) : sharedCols;

  useEffect(() => {
    if (hasShared && !fileData && numericalCols.length === 0 && textCols.length === 0 && activeData?.length) {
      const first = activeData[0];
      setNumericalCols(columns.filter(c => typeof first[c] === "number"));
      setTextCols(columns.filter(c => typeof first[c] === "string"));
    }
  }, [hasShared, fileData, activeData, columns]);

  async function onFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setResult(null); setError("");
    try {
      const parsed = await parseFile(f);
      setFileData(parsed?.allData || null);
      if (parsed?.allData?.length) {
        const first = parsed.allData[0];
        const cols = Object.keys(first);
        setNumericalCols(cols.filter(c => typeof first[c] === "number"));
        setTextCols(cols.filter(c => typeof first[c] === "string"));
      }
    } catch (err) { setError("Ошибка чтения файла: " + err.message); }
  }

  async function onRun() {
    if (!activeData) return;
    setBusy(true); setError(""); setResult(null);
    try {
      const res = await createFeatureCombinations(activeData, numericalCols.length > 0 ? numericalCols : null, textCols.length > 0 ? textCols : null, maxPairs);
      setResult(res);
    } catch (err) { setError("Ошибка: " + err.message); }
    finally { setBusy(false); }
  }

  return (
    <div className="card">
      <h2>🔗 Объединение признаков</h2>
      <p className="muted">Создание новых признаков: суммы, разности, отношения, произведения, конкатенации.</p>

      <div style={{ marginBottom: 12 }}>
        {hasShared && !fileData && (
          <div className="ok" style={{ padding: 8, borderRadius: 6, background: "var(--bg-secondary)", marginBottom: 8 }}>
            ✅ Данные с главной: <b>{activeData.length} строк</b>, {columns.length} колонок
          </div>
        )}
        <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileChange} />
      </div>

      {columns.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 12 }}>
          <div>
            <label><b>🔢 Числовые:</b></label>
            <div style={{ maxHeight: 200, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 6, padding: 8, marginTop: 4, background: "var(--bg-secondary)" }}>
              {columns.filter(c => typeof activeData?.[0]?.[c] === "number").map(c => (
                <label key={c} style={{ display: "flex", alignItems: "center", gap: 6, padding: "2px 0", fontSize: 13 }}>
                  <input type="checkbox" checked={numericalCols.includes(c)} onChange={e => setNumericalCols(p => e.target.checked ? [...p, c] : p.filter(x => x !== c))} />{c}
                </label>
              ))}
            </div>
          </div>
          <div>
            <label><b>📝 Текстовые:</b></label>
            <div style={{ maxHeight: 200, overflowY: "auto", border: "1px solid var(--border)", borderRadius: 6, padding: 8, marginTop: 4, background: "var(--bg-secondary)" }}>
              {columns.filter(c => typeof activeData?.[0]?.[c] === "string").map(c => (
                <label key={c} style={{ display: "flex", alignItems: "center", gap: 6, padding: "2px 0", fontSize: 13 }}>
                  <input type="checkbox" checked={textCols.includes(c)} onChange={e => setTextCols(p => e.target.checked ? [...p, c] : p.filter(x => x !== c))} />{c}
                </label>
              ))}
            </div>
          </div>
        </div>
      )}

      <div style={{ marginBottom: 12 }}>
        <label><b>Макс. новых признаков:</b> {maxPairs}
          <input type="range" min="5" max="50" step="1" value={maxPairs} onChange={e => setMaxPairs(Number(e.target.value))} style={{ width: "100%", marginTop: 4 }} />
        </label>
      </div>

      <button className="primary" onClick={onRun} disabled={!activeData || busy}>{busy ? "⏳ Создание..." : "🔗 Создать комбинации"}</button>
      {error && <p className="error" style={{ marginTop: 8 }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 12 }}>
          <p className="ok">✅ Создано <b>{result.n_new}</b> новых признаков (всего: {result.total_columns})</p>
          {result.new_columns?.length > 0 && (
            <details open><summary>Новые колонки ({result.new_columns.length})</summary>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 8 }}>
                {result.new_columns.map(c => <span key={c} className="tag" style={{ background: "var(--primary)", color: "#fff", padding: "2px 8px", borderRadius: 12, fontSize: 12 }}>{c}</span>)}
              </div>
            </details>
          )}
          {result.data?.length > 0 && (
            <details><summary>Превью (первые 10)</summary>
              <div className="table-wrap" style={{ marginTop: 8, maxHeight: 400, overflowY: "auto" }}>
                <table><thead><tr>{Object.keys(result.data[0]).map(h => <th key={h} style={{ fontSize: 11 }}>{h}</th>)}</tr></thead>
                  <tbody>{result.data.slice(0, 10).map((row, i) => (<tr key={i}>{Object.values(row).map((v, j) => (<td key={j} style={{ fontSize: 11 }}>{v !== null && v !== undefined ? String(v) : ""}</td>))}</tr>))}</tbody></table>
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
