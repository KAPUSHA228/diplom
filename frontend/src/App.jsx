import { useEffect, useMemo, useState } from "react";
import {
    getTaskStatus,
    healthcheck,
    uploadForCorrelation,
    uploadForShap,
    uploadForTrain } from "./api";

// Роутер будет добавлен когда появится Shell-приложение
// import { BrowserRouter, Routes, Route } from "react-router-dom";

function PollingTask({ taskId, title }) {
  const [task, setTask] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!taskId) return;
    let stopped = false;
    const timer = setInterval(async () => {
      try {
        const status = await getTaskStatus(taskId);
        if (!stopped) setTask(status);
        if (status.status === "completed" || status.status === "failed") {
          clearInterval(timer);
        }
      } catch (e) {
        if (!stopped) setError(String(e.message || e));
        clearInterval(timer);
      }
    }, 2000);
    return () => {
        stopped = true;
        clearInterval(timer);
      };
    }, [taskId]);
    if (!taskId) return null;
    return (
      <div className="card">
        <h3>{title}</h3>
        <p>
          <b>Task ID:</b> {taskId}
        </p>
        {error ? <p className="error">{error}</p> : null}
        {task ? (
          <>
            <p>
              <b>Статус:</b> {task.status}
            </p>
            <p>
            <b>Этап:</b> {task.stage || "-"} | <b>Прогресс:</b> {task.progress ?? 0}%
          </p>
          {task.result ? <pre>{JSON.stringify(task.result, null, 2)}</pre> : null}
          {task.error ? <pre className="error">{task.error}</pre> : null}
        </>
      ) : (
        <p>Ожидание обновлений...</p>
      )}
    </div>
  );
}
export default function App() {
    const [health, setHealth] = useState("checking");
    const [file, setFile] = useState(null);
    const [csvPreview, setCsvPreview] = useState({ headers: [], rows: [], rowCount: 0, riskPct: null });
    const [corrResult, setCorrResult] = useState(null);
    const [trainTaskId, setTrainTaskId] = useState("");
    const [shapTaskId, setShapTaskId] = useState("");
    const [error, setError] = useState("");
    const [busy, setBusy] = useState(false);
    const canRun = useMemo(() => !!file && !busy, [file, busy]);
    useEffect(() => {
      healthcheck()
        .then(() => setHealth("ok"))
        .catch(() => setHealth("down"));
    }, []);
    async function onTrain() {
        if (!file) return;
        try {
          setBusy(true);
          setError("");
          const res = await uploadForTrain(file);
          setTrainTaskId(res.task_id);
        } catch (e) {
          setError(String(e.message || e));
        } finally {
          setBusy(false);
        }
      }

      async function onCorrelation() {
        if (!file) return;
        try {
          setBusy(true);
          setError("");
          const res = await uploadForCorrelation(file);
          setCorrResult(res);
        } catch (e) {
          setError(String(e.message || e));
        } finally {
          setBusy(false);
        }
      }

      async function onShap() {
        if (!file) return;
        try {
          setBusy(true);
          setError("");
          const res = await uploadForShap(file, "XGB");
          setShapTaskId(res.task_id);
        } catch (e) {
          setError(String(e.message || e));
        } finally {
          setBusy(false);
        }
      }
      async function onFileChange(e) {
        const next = e.target.files?.[0] || null;
        setFile(next);
        setCorrResult(null);
        if (!next) {
          setCsvPreview({ headers: [], rows: [], rowCount: 0, riskPct: null });
          return;
        }
        try {
          const text = await next.text();
          const lines = text.split(/\r?\n/).filter(Boolean);
          if (lines.length === 0) return;
          const headers = lines[0].split(",");
          const rows = lines.slice(1, 6).map((line) => line.split(","));
          const riskIdx = headers.indexOf("risk_flag");
          let riskPct = null;
          if (riskIdx >= 0) {
            const vals = lines.slice(1).map((line) => Number((line.split(",")[riskIdx] || "").trim()));
            const valid = vals.filter((v) => Number.isFinite(v));
            if (valid.length) {
              const risky = valid.filter((v) => v === 1).length;
              riskPct = (risky / valid.length) * 100;
            }
          }
          setCsvPreview({
            headers,
            rows,
            rowCount: Math.max(lines.length - 1, 0),
            riskPct
          });
        } catch {
          // preview optional
          setCsvPreview({ headers: [], rows: [], rowCount: 0, riskPct: null });
        }
      }

      function renderCorrelationTable() {
        if (!corrResult?.correlation_matrix) return null;
        const matrix = corrResult.correlation_matrix;
        const cols = Object.keys(matrix);
        const getCellClass = (v) => {
          if (v > 0.7) return "heat-high";
          if (v > 0.3) return "heat-mid";
          if (v < -0.7) return "heat-neg-high";
          if (v < -0.3) return "heat-neg-mid";
          return "heat-low";
        };
        return (
          <div className="table-wrap">
            <table className="matrix">
              <thead>
                <tr>
                  <th>feature</th>
                  {cols.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
            </thead>
          <tbody>
            {cols.map((r) => (
              <tr key={r}>
                <td><b>{r}</b></td>
                {cols.map((c) => {
                  const v = Number(matrix[r]?.[c] ?? 0);
                  return (
                    <td key={`${r}_${c}`} className={getCellClass(v)}>
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

    return (
        <main className="container">
          <h1>Student Risk Platform (React + FastAPI)</h1>
          <p className="muted">
        React — основной интерфейс. Streamlit можно оставить как legacy/debug.
        </p>
        <p>
            Состояние API:{" "}
            <span className={health === "ok" ? "ok" : health === "down" ? "error" : ""}>{health}</span>
          </p>
          <div className="card">
            <h2>1) Загрузка CSV</h2>
            <input
              type="file"
              accept=".csv"
              onChange={onFileChange}
            />
            <p>Файл: {file?.name || "не выбран"}</p>
            {csvPreview.rowCount > 0 ? (
                <>
                <p>
                  Строк: <b>{csvPreview.rowCount}</b>
                  {csvPreview.riskPct != null ? (
                    <>
                      {" "} | Доля `risk_flag=1`: <b>{csvPreview.riskPct.toFixed(1)}%</b>
                    </>
                  ) : null}
                </p>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        {csvPreview.headers.map((h) => (
                          <th key={h}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                    {csvPreview.rows.map((r, i) => (
                    <tr key={i}>
                      {r.map((cell, j) => (
                        <td key={`${i}_${j}`}>{cell}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </div>
      <div className="card">
        <h2>2) Корреляционный анализ</h2>

            <div className="row">
            <button onClick={onCorrelation} disabled={!canRun}>
            Запустить корреляцию
          </button>
        </div>
        {corrResult ? (
          <>
            <p>
              Размер данных: <b>{corrResult.n_rows}</b> строк, <b>{corrResult.n_columns}</b> колонок
            </p>
            {renderCorrelationTable()}
          </>
        ) : (
          <p className="muted">После запуска здесь появится корреляционная матрица.</p>
        )}
      </div>
      <div className="card">
        <h2>3) Асинхронные ML-задачи (RQ)</h2>
        <div className="row">
              <button onClick={onTrain} disabled={!canRun}>
                Запустить обучение (RQ)
              </button>
              <button onClick={onShap} disabled={!canRun}>
                Запустить SHAP (RQ)
              </button>
              </div>
        {error ? <p className="error">{error}</p> : null}
      </div>
      <PollingTask taskId={trainTaskId} title="3) Обучение модели" />
      <PollingTask taskId={shapTaskId} title="4) SHAP объяснения" />
    </main>
  );
}