import { useState, useEffect } from "react";
import { createCompositeScore } from "../api";
import { parseFile } from "../utils/parseFile";
import { useSharedData } from "../hooks/useSharedData";

export default function CompositeScore() {
  const { data: sharedData, columns: sharedCols, hasShared } = useSharedData();
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [weights, setWeights] = useState({});
  const [scoreName, setScoreName] = useState("custom_score");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const activeData = fileData || sharedData;
  const numericCols = activeData?.length ? Object.keys(activeData[0]).filter(k => typeof activeData[0][k] === "number") : [];

  useEffect(() => {
    if (hasShared && !fileData && Object.keys(weights).length === 0) {
      const w = {};
      numericCols.forEach(c => { w[c] = 1; });
      setWeights(w);
    }
  }, [hasShared, fileData, numericCols]);

  async function onFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setResult(null);
    try {
      const parsed = await parseFile(f);
      setFileData(parsed?.allData || null);
      if (parsed?.allData?.length) {
        const w = {};
        Object.keys(parsed.allData[0]).filter(k => typeof parsed.allData[0][k] === "number").forEach(c => { w[c] = 1; });
        setWeights(w);
      }
    } catch { /* ignore */ }
  }

  async function onRun() {
    if (!activeData || Object.keys(weights).length === 0) return;
    setBusy(true); setError("");
    try {
      const cleanWeights = {};
      Object.entries(weights).forEach(([k, v]) => { cleanWeights[k] = Number(v) || 0; });
      const res = await createCompositeScore(activeData, cleanWeights, scoreName);
      setResult(res);
    } catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  return (
    <div className="card">
      <h2>🎯 Конструктор композитных оценок</h2>

      <div style={{ marginBottom: 12 }}>
        {hasShared && !fileData && (
          <div className="ok" style={{ padding: 8, borderRadius: 6, background: "var(--bg-secondary)", marginBottom: 8 }}>
            ✅ Данные с главной: <b>{sharedData.length} строк</b>, {numericCols.length} числовых колонок
          </div>
        )}
        <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileChange} />
      </div>

      {numericCols.length > 0 && (
        <>
          <label><b>Название оценки:</b>
            <input type="text" value={scoreName} onChange={e => setScoreName(e.target.value)} style={{ marginLeft: 8, padding: 4, borderRadius: 4, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)" }} />
          </label>

          <div style={{ marginTop: 12 }}>
            <label><b>Веса признаков:</b></label>
            <div style={{ maxHeight: 300, overflowY: "auto", marginTop: 8 }}>
              {numericCols.map(col => (
                <div key={col} style={{ marginBottom: 6, display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ minWidth: 140, fontSize: 13 }}>{col}</span>
                  <input type="range" min="-3" max="3" step="1" value={weights[col] || 0} onChange={e => setWeights(p => ({ ...p, [col]: Number(e.target.value) }))} style={{ flex: 1 }} />
                  <span style={{ minWidth: 20, textAlign: "center" }}>{weights[col] || 0}</span>
                </div>
              ))}
            </div>
          </div>

          <button className="primary" onClick={onRun} disabled={busy} style={{ marginTop: 12 }}>Создать оценку</button>
        </>
      )}

      {error && <p className="error">{error}</p>}
      {result && (
        <div style={{ marginTop: 12 }}>
          <p className="ok">✅ Оценка <b>{result.score_name}</b> создана</p>
          {result.statistics && (
            <table className="matrix"><tbody>
              {Object.entries(result.statistics).map(([k, v]) => <tr key={k}><td><b>{k}</b></td><td>{typeof v === "number" ? v.toFixed(2) : v}</td></tr>)}
            </tbody></table>
          )}
        </div>
      )}
    </div>
  );
}
