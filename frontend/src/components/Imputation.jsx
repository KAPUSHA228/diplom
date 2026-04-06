import { useState } from "react";
import { handleImputation } from "../api";

export default function Imputation() {
  const [file, setFile] = useState(null);
  const [strategy, setStrategy] = useState("auto");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onRun() {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const text = await file.text();
      const lines = text.split(/\r?\n/).filter(Boolean);
      const headers = lines[0].split(",");
      const rows = lines.slice(1).map((l) => l.split(","));
      const data = rows.map((r) => {
        const obj = {};
        headers.forEach((h, i) => { obj[h.trim()] = r[i]?.trim() || ""; });
        return obj;
      });
      const res = await handleImputation(data, strategy);
      setResult(res);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>Обработка пропусков и выбросов</h2>
      <input type="file" accept=".csv" onChange={(e) => { setFile(e.target.files?.[0] || null); setResult(null); }} />
      <div className="row" style={{ marginTop: 8 }}>
        <label>Стратегия: </label>
        <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
          <option value="auto">Auto</option>
          <option value="fill_median">Медиана</option>
          <option value="fill_mean">Среднее</option>
          <option value="interpolate">Интерполяция</option>
          <option value="drop_rows">Удалить строки</option>
        </select>
        <button onClick={onRun} disabled={busy || !file}>Обработать</button>
      </div>
      {error && <p className="error">{error}</p>}
      {result && (
        <div>
          <p>Обработано: <b>{result.report?.final_shape?.[0] || "?"}</b> строк, <b>{result.report?.final_shape?.[1] || "?"}</b> колонок</p>
          {result.report?.actions?.map((a, i) => (
            <p key={i}>• {a.column}: {a.message}</p>
          ))}
          {Object.keys(result.outliers || {}).length > 0 && (
            <details>
              <summary>Выбросы ({Object.keys(result.outliers).length} колонок)</summary>
              {Object.entries(result.outliers).map(([col, info]) => (
                <p key={col}><b>{col}:</b> {info.n_outliers} выбросов ({info.percentage}%)</p>
              ))}
            </details>
          )}
        </div>
      )}
    </div>
  );
}
