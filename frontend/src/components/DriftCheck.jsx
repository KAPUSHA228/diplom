import { useState } from "react";
import { checkDrift } from "../api";

export default function DriftCheck() {
  const [refFile, setRefFile] = useState(null);
  const [curFile, setCurFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onRun() {
    if (!refFile || !curFile) return;
    setBusy(true);
    setError("");
    try {
      const parse = async (f) => {
        const text = await f.text();
        const lines = text.split(/\r?\n/).filter(Boolean);
        const headers = lines[0].split(",");
        return lines.slice(1).map((l) => {
          const r = l.split(",");
          const obj = {};
          headers.forEach((h, i) => { obj[h.trim()] = r[i]?.trim() || ""; });
          return obj;
        });
      };
      const ref = await parse(refFile);
      const cur = await parse(curFile);
      const res = await checkDrift(ref, cur);
      setResult(res);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>Проверка дрейфа данных</h2>
      <p>Загрузите эталонные и новые данные для сравнения распределений</p>
      <div className="row">
        <div>
          <label>Эталон: </label>
          <input type="file" accept=".csv" onChange={(e) => setRefFile(e.target.files?.[0] || null)} />
        </div>
        <div>
          <label>Новые данные: </label>
          <input type="file" accept=".csv" onChange={(e) => setCurFile(e.target.files?.[0] || null)} />
        </div>
      </div>
      <button onClick={onRun} disabled={busy || !refFile || !curFile}>Проверить дрейф</button>
      {error && <p className="error">{error}</p>}
      {result && (
        <div>
          <p>
            Статус: {result.overall_drift ? <span className="error">🔴 Есть дрейф</span> : <span className="ok">🟢 Нет дрейфа</span>}
          </p>
          <p>Процент дрейфа: <b>{result.drift_percentage}%</b></p>
          {result.drifted_features?.length > 0 && (
            <p>Дрейфующие признаки: {result.drifted_features.join(", ")}</p>
          )}
          {result.recommendations?.length > 0 && (
            <details>
              <summary>Рекомендации</summary>
              {result.recommendations.map((r, i) => <p key={i}>• {r}</p>)}
            </details>
          )}
        </div>
      )}
    </div>
  );
}
