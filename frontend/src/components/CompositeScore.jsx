import { useState } from "react";
import { createCompositeScore } from "../api";

export default function CompositeScore() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [weights, setWeights] = useState({});
  const [scoreName, setScoreName] = useState("custom_score");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onFileChange(e) {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setResult(null);
    if (!f) return;
    try {
      const text = await f.text();
      const lines = text.split(/\r?\n/).filter(Boolean);
      const headers = lines[0].split(",").map((h) => h.trim());
      const rows = lines.slice(1).map((l) => l.split(","));
      const parsed = rows.map((r) => {
        const obj = {};
        headers.forEach((h, i) => {
          const v = r[i]?.trim() || "";
          obj[h] = isNaN(v) ? v : Number(v);
        });
        return obj;
      });
      setData(parsed);
      const numericCols = headers.filter((h) => typeof parsed[0]?.[h] === "number");
      const w = {};
      numericCols.forEach((c) => { w[c] = 0; });
      setWeights(w);
    } catch { /* ignore */ }
  }

  function onWeightChange(col, val) {
    setWeights((prev) => ({ ...prev, [col]: val }));
  }

  async function onRun() {
    if (!data) return;
    const active = {};
    Object.entries(weights).forEach(([k, v]) => { if (Math.abs(v) > 1e-6) active[k] = v; });
    if (Object.keys(active).length === 0) { setError("Укажите хотя бы один ненулевой вес"); return; }
    setBusy(true); setError("");
    try {
      const res = await createCompositeScore(data, active, scoreName);
      setResult(res);
    } catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  return (
    <div className="card">
      <h2>Конструктор композитных оценок</h2>
      <input type="file" accept=".csv" onChange={onFileChange} />
      {data && (
        <>
          <div className="row" style={{ marginTop: 8 }}>
            <label>Название: </label>
            <input value={scoreName} onChange={(e) => setScoreName(e.target.value)} />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginTop: 8 }}>
            {Object.entries(weights).map(([col, val]) => (
              <label key={col} style={{ fontSize: 13 }}>
                {col}:{" "}
                <input type="range" min="-3" max="3" step="0.1" value={val}
                  onChange={(e) => onWeightChange(col, parseFloat(e.target.value))} />
                <span>{val.toFixed(1)}</span>
              </label>
            ))}
          </div>
          <button onClick={onRun} disabled={busy}>Создать оценку</button>
          {error && <p className="error">{error}</p>}
          {result && (
            <div>
              <p>Создана оценка: <b>{result.score_name}</b></p>
              {result.statistics && (
                <div className="table-wrap">
                  <table>
                    <tbody>
                      {Object.entries(result.statistics).map(([k, v]) => (
                        <tr key={k}><td><b>{k}</b></td><td>{typeof v === "number" ? v.toFixed(2) : v}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
