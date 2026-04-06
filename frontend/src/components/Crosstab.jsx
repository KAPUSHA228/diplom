import { useState } from "react";
import { buildCrosstab } from "../api";

export default function Crosstab() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [rowVar, setRowVar] = useState("");
  const [colVar, setColVar] = useState("");
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
        headers.forEach((h, i) => { obj[h] = r[i]?.trim() || ""; });
        return obj;
      });
      setData(parsed);
      if (headers.length >= 2) { setRowVar(headers[0]); setColVar(headers[1]); }
    } catch { /* ignore */ }
  }

  async function onRun() {
    if (!data || !rowVar || !colVar) return;
    setBusy(true);
    setError("");
    try {
      const res = await buildCrosstab(data, rowVar, colVar);
      setResult(res);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>Кросс-таблица (χ²-тест)</h2>
      <input type="file" accept=".csv" onChange={onFileChange} />
      {data && (
        <>
          <div className="row" style={{ marginTop: 8 }}>
            <label>Строки: </label>
            <select value={rowVar} onChange={(e) => setRowVar(e.target.value)}>
              {Object.keys(data[0]).map((h) => <option key={h} value={h}>{h}</option>)}
            </select>
            <label>Столбцы: </label>
            <select value={colVar} onChange={(e) => setColVar(e.target.value)}>
              {Object.keys(data[0]).map((h) => <option key={h} value={h}>{h}</option>)}
            </select>
            <button onClick={onRun} disabled={busy}>Построить</button>
          </div>
          {error && <p className="error">{error}</p>}
          {result?.table && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th></th>
                    {Object.keys(result.table).map((c) => <th key={c}>{c}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.table).map(([row, vals]) => (
                    <tr key={row}>
                      <td><b>{row}</b></td>
                      {Object.values(vals).map((v, i) => <td key={i}>{typeof v === "number" ? v.toFixed(2) : v}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {result?.chi2_test && (
            <p>
              χ² тест: p-value = <b>{result.chi2_test.p_value?.toFixed(4)}</b>
              {" "}{result.chi2_test.significant ? "✅ Значимо" : "❌ Не значимо"}
            </p>
          )}
        </>
      )}
    </div>
  );
}
