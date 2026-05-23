import { useState } from "react";
import { handleImputation } from "../api";
import { parseFile } from "../utils/parseFile";
import { useDatasetLoader } from "../hooks/useDatasetLoader";

const STRATEGIES = [
  { value: "auto", label: "Авто" },
  { value: "median", label: "Медиана" },
  { value: "mean", label: "Среднее" },
  { value: "interpolate", label: "Интерполяция" },
  { value: "drop_rows", label: "Удалить строки" },
];
const OUTLIER_METHODS = [
  { value: "iqr", label: "IQR (межквартильный размах)", defaultThreshold: 1.5 },
  {
    value: "zscore",
    label: "Z‑score (стандартное отклонение)",
    defaultThreshold: 3.0,
  },
];

export default function Imputation() {
  const {
    data: sharedData,
    columns: sharedCols,
    hasShared,
    loading,
  } = useDatasetLoader();
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [strategy, setStrategy] = useState("auto");
  const [columnDropThreshold, setColumnDropThreshold] = useState(30);
  const [outlierMethod, setOutlierMethod] = useState("iqr");
  const [outlierThreshold, setOutlierThreshold] = useState(1.5);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const activeData = fileData || sharedData;

  if (loading) {
    return <div className="card">Загрузка данных...</div>;
  }

  async function onFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError("");
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
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const res = await handleImputation(
        activeData,
        strategy,
        columnDropThreshold,
        outlierMethod,
        outlierThreshold,
      );
      setResult(res);
    } catch (err) {
      setError("Ошибка: " + err.message);
    } finally {
      setBusy(false);
    }
  }

  const handleMethodChange = (method) => {
    setOutlierMethod(method);
    const def =
      OUTLIER_METHODS.find((m) => m.value === method)?.defaultThreshold || 1.5;
    setOutlierThreshold(def);
  };

  return (
    <div className="card">
      <h2>Обработка пропусков и выбросов</h2>

      {/* Данные */}
      <div style={{ marginBottom: 12 }}>
        {hasShared && !fileData && (
          <div
            className="ok"
            style={{
              padding: 8,
              borderRadius: 6,
              background: "var(--bg-secondary)",
              marginBottom: 8,
            }}
          >
            Используются данные с главной вкладки:{" "}
            <b>{sharedData.length} строк</b>, {sharedCols.length} колонок
          </div>
        )}
        <label>
          <b>Источник данных:</b>
        </label>
        <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileChange} />
        {fileData && (
          <span className="ok" style={{ marginLeft: 8 }}>
            Загружено: {fileData.length} строк
          </span>
        )}
        {file && (
          <span className="ok" style={{ marginLeft: 8 }}>
            {file.name}
          </span>
        )}
      </div>

      {/* Стратегия */}
      <div style={{ marginBottom: 12 }}>
        <label>
          <b>Стратегия обработки пропусков:</b>
        </label>
        <div className="strategy-grid" style={{ marginTop: 8 }}>
          {STRATEGIES.map((s) => (
            <label
              key={s.value}
              className={`strategy-option ${strategy === s.value ? "selected" : ""}`}
            >
              <input
                type="radio"
                name="imputation-strategy"
                value={s.value}
                checked={strategy === s.value}
                onChange={() => setStrategy(s.value)}
              />
              {s.label}
            </label>
          ))}
        </div>
      </div>

      {/* Дополнительные параметры для стратегии Auto: порог удаления столбца */}
      {strategy === "auto" && (
        <div style={{ marginBottom: 12 }}>
          <label>
            <b>Порог удаления столбца (% пропусков):</b> {columnDropThreshold}
            <input
              type="range"
              min="10"
              max="70"
              step="5"
              value={columnDropThreshold}
              onChange={(e) => setColumnDropThreshold(parseInt(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>
          <p className="muted" style={{ fontSize: 12 }}>
            Если доля пропусков в столбце превышает этот порог, столбец будет
            удалён.
          </p>
        </div>
      )}

      {/* Детекция выбросов – выбор метода и порога */}
      <div style={{ marginBottom: 12 }}>
        <label>
          <b>Метод детекции выбросов:</b>
        </label>
        <div className="strategy-grid" style={{ marginTop: 8 }}>
          {OUTLIER_METHODS.map((m) => (
            <label
              key={m.value}
              className={`strategy-option ${outlierMethod === m.value ? "selected" : ""}`}
            >
              <input
                type="radio"
                name="outlier-method"
                value={m.value}
                checked={outlierMethod === m.value}
                onChange={() => handleMethodChange(m.value)}
              />
              {m.label}
            </label>
          ))}
        </div>
        <div style={{ marginTop: 8 }}>
          <label>
            <b>Порог чувствительности:</b> {outlierThreshold}
            <input
              type="range"
              min="0.5"
              max="5.0"
              step="0.1"
              value={outlierThreshold}
              onChange={(e) => setOutlierThreshold(parseFloat(e.target.value))}
              style={{ width: "100%" }}
            />
          </label>
          <p className="muted" style={{ fontSize: 12 }}>
            {outlierMethod === "iqr"
              ? "IQR: значение > Q₃ + k·IQR или < Q₁ − k·IQR считается выбросом (обычно k=1.5)."
              : "Z‑score: значение, отклоняющееся более чем на k стандартных отклонений, считается выбросом (обычно k=3.0)."}
          </p>
        </div>
      </div>

      {/* Запуск */}
      <button
        className="primary"
        onClick={onRun}
        disabled={busy || !activeData}
      >
        {busy ? "Обработка..." : "Обработать"}
      </button>

      {error && (
        <p className="error" style={{ marginTop: 8 }}>
          {error}
        </p>
      )}

      {/* Результат */}
      {result && (
        <div style={{ marginTop: 12 }}>
          <p>
            Обработано: <b>{result.report?.final_shape?.[0] || "?"}</b> строк,{" "}
            <b>{result.report?.final_shape?.[1] || "?"}</b> колонок
          </p>
          <p className="muted" style={{ fontSize: 12 }}>
            Выбросы обнаружены методом{" "}
            <b>{result.outlier_method === "iqr" ? "IQR" : "Z‑score"}</b> (порог
            = {result.outlier_threshold})<br />
            {strategy === "auto" &&
              `Порог удаления столбцов по пропускам: ${columnDropThreshold}%`}
          </p>
          {result.report?.actions?.length > 0 && (
            <details open>
              <summary>Действия ({result.report.actions.length})</summary>
              {result.report.actions.map((a, i) => (
                <p key={i}>
                  • <b>{a.column}</b>: {a.message}
                </p>
              ))}
            </details>
          )}
          {Object.keys(result.outliers || {}).length > 0 && (
            <details>
              <summary>
                Выбросы ({Object.keys(result.outliers).length} колонок)
              </summary>
              {Object.entries(result.outliers).map(([col, info]) => (
                <p key={col}>
                  <b>{col}:</b> {info.n_outliers} выбросов (
                  {info.percentage?.toFixed(1)}%)
                </p>
              ))}
            </details>
          )}
        </div>
      )}
    </div>
  );
}
