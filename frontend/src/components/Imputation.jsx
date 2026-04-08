import { useState } from "react";
import { handleImputation } from "../api";
import { parseFile } from "../utils/parseFile";
import { useSharedData } from "../hooks/useSharedData";

const STRATEGIES = [
  { value: "auto", label: "🤖 Авто" },
  { value: "median", label: "📊 Медиана" },
  { value: "mean", label: "🔢 Среднее" },
  { value: "interpolate", label: "↕️ Интерполяция" },
  { value: "drop_rows", label: "🗑️ Удалить строки" },
];

export default function Imputation() {
  const { data: sharedData, columns: sharedCols, hasShared } = useSharedData();
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [strategy, setStrategy] = useState("auto");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const activeData = fileData || sharedData;

  async function onFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setResult(null); setError("");
    try {
      const parsed = await parseFile(f);
      setFileData(parsed?.allData || null);
    } catch (err) {
      setError("Ошибка чтения файла: " + err.message);
      setFileData(null);
    }
  }

  async function onRun() {
    if (!activeData || !activeData.length) return;
    setBusy(true); setError(""); setResult(null);
    try {
      const res = await handleImputation(activeData, strategy);
      setResult(res);
    } catch (err) {
      setError("Ошибка: " + err.message);
    } finally { setBusy(false); }
  }

  return (
    <div className="card">
      <h2>🔧 Обработка пропусков и выбросов</h2>

      {/* Данные */}
      <div style={{ marginBottom: 12 }}>
        {hasShared && !fileData && (
          <div className="ok" style={{ padding: 8, borderRadius: 6, background: "var(--bg-secondary)", marginBottom: 8 }}>
            ✅ Используются данные с главной вкладки: <b>{sharedData.length} строк</b>, {sharedCols.length} колонок
          </div>
        )}
        <label><b>Источник данных:</b></label>
        <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileChange} />
        {fileData && <span className="ok" style={{ marginLeft: 8 }}>✅ Загружено: {fileData.length} строк</span>}
      </div>

      {/* Стратегия */}
      <div style={{ marginBottom: 12 }}>
        <label><b>Стратегия:</b></label>
        <div className="strategy-grid" style={{ marginTop: 8 }}>
          {STRATEGIES.map(s => (
            <label key={s.value} className={`strategy-option ${strategy === s.value ? "selected" : ""}`}>
              <input type="radio" name="imputation-strategy" value={s.value} checked={strategy === s.value} onChange={() => setStrategy(s.value)} />
              {s.label}
            </label>
          ))}
        </div>
      </div>

      {/* Запуск */}
      <button className="primary" onClick={onRun} disabled={busy || !activeData}>
        {busy ? "⏳ Обработка..." : "🔧 Обработать"}
      </button>

      {error && <p className="error" style={{ marginTop: 8 }}>{error}</p>}

      {/* Результат */}
      {result && (
        <div style={{ marginTop: 12 }}>
          <p>Обработано: <b>{result.report?.final_shape?.[0] || "?"}</b> строк, <b>{result.report?.final_shape?.[1] || "?"}</b> колонок</p>
          {result.report?.actions?.length > 0 && (
            <details open>
              <summary>Действия ({result.report.actions.length})</summary>
              {result.report.actions.map((a, i) => (
                <p key={i}>• <b>{a.column}</b>: {a.message}</p>
              ))}
            </details>
          )}
          {Object.keys(result.outliers || {}).length > 0 && (
            <details>
              <summary>Выбросы ({Object.keys(result.outliers).length} колонок)</summary>
              {Object.entries(result.outliers).map(([col, info]) => (
                <p key={col}><b>{col}:</b> {info.n_outliers} выбросов ({info.percentage?.toFixed(1)}%)</p>
              ))}
            </details>
          )}
        </div>
      )}
    </div>
  );
}
