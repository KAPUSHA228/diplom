import { useEffect, useMemo, useState } from "react";
import { HashRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import Plot from "react-plotly.js";
import {
  getTaskStatus,
  healthcheck,
  uploadForCorrelation,
  uploadForShap,
  uploadForTrain,
} from "./api";
import Imputation from "./components/Imputation";
import DriftCheck from "./components/DriftCheck";
import Crosstab from "./components/Crosstab";
import TimeSeries from "./components/TimeSeries";
import CompositeScore from "./components/CompositeScore";
import Experiments from "./components/Experiments";
import "./styles.css";

const NAV = [
  { path: "/", label: "📊 Главное" },
  { path: "/imputation", label: "🔧 Пропуски" },
  { path: "/crosstab", label: "📈 Кросс-таблицы" },
  { path: "/timeseries", label: "📉 Временные ряды" },
  { path: "/composite", label: "🎯 Композитные оценки" },
  { path: "/drift", label: "🔄 Дрейф" },
  { path: "/experiments", label: "📁 Эксперименты" },
];

function Tabs() {
  const location = useLocation();
  return (
    <div className="tabs">
      {NAV.map((n) => (
        <Link key={n.path} to={n.path} className={location.pathname === n.path ? "active" : ""}>
          {n.label}
        </Link>
      ))}
    </div>
  );
}

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
        if (status.status === "completed" || status.status === "failed") clearInterval(timer);
      } catch (e) {
        if (!stopped) setError(String(e.message || e));
        clearInterval(timer);
      }
    }, 2000);
    return () => { stopped = true; clearInterval(timer); };
  }, [taskId]);

  if (!taskId) return null;
  return (
    <div className="card">
      <h3>{title}</h3>
      <p><b>Task ID:</b> {taskId}</p>
      {error ? <p className="error">{error}</p> : null}
      {task ? (
        <>
          <p><b>Статус:</b> {task.status}</p>
          <p><b>Этап:</b> {task.stage || "-"} | <b>Прогресс:</b> {task.progress ?? 0}%</p>
          {task.result ? <pre>{JSON.stringify(task.result, null, 2)}</pre> : null}
          {task.error ? <pre className="error">{task.error}</pre> : null}
        </>
      ) : <p>Ожидание обновлений...</p>}
    </div>
  );
}

function MainPage() {
  const [file, setFile] = useState(null);
  const [csvPreview, setCsvPreview] = useState({ headers: [], rows: [], rowCount: 0, riskPct: null });
  const [corrResult, setCorrResult] = useState(null);
  const [trainTaskId, setTrainTaskId] = useState("");
  const [shapTaskId, setShapTaskId] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const canRun = useMemo(() => !!file && !busy, [file, busy]);

  async function onTrain() {
    if (!file) return;
    try { setBusy(true); setError(""); const res = await uploadForTrain(file); setTrainTaskId(res.task_id); }
    catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  async function onCorrelation() {
    if (!file) return;
    try { setBusy(true); setError(""); const res = await uploadForCorrelation(file); setCorrResult(res); }
    catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  async function onShap() {
    if (!file) return;
    try { setBusy(true); setError(""); const res = await uploadForShap(file, "XGB"); setShapTaskId(res.task_id); }
    catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  async function onFileChange(e) {
    const next = e.target.files?.[0] || null;
    setFile(next); setCorrResult(null);
    if (!next) { setCsvPreview({ headers: [], rows: [], rowCount: 0, riskPct: null }); return; }
    try {
      const text = await next.text();
      const lines = text.split(/\r?\n/).filter(Boolean);
      if (lines.length === 0) return;
      const headers = lines[0].split(",");
      const rows = lines.slice(1, 6).map((l) => l.split(","));
      const riskIdx = headers.indexOf("risk_flag");
      let riskPct = null;
      if (riskIdx >= 0) {
        const vals = lines.slice(1).map((l) => Number((l.split(",")[riskIdx] || "").trim()));
        const valid = vals.filter((v) => Number.isFinite(v));
        if (valid.length) { const risky = valid.filter((v) => v === 1).length; riskPct = (risky / valid.length) * 100; }
      }
      setCsvPreview({ headers, rows, rowCount: Math.max(lines.length - 1, 0), riskPct });
    } catch { setCsvPreview({ headers: [], rows: [], rowCount: 0, riskPct: null }); }
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
          <thead><tr><th>feature</th>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {cols.map((r) => (
              <tr key={r}>
                <td><b>{r}</b></td>
                {cols.map((c) => {
                  const v = Number(matrix[r]?.[c] ?? 0);
                  return <td key={`${r}_${c}`} className={getCellClass(v)}>{v.toFixed(2)}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <>
      <div className="card">
        <h2>1) Загрузка CSV</h2>
        <input type="file" accept=".csv" onChange={onFileChange} />
        <p>Файл: {file?.name || "не выбран"}</p>
        {csvPreview.rowCount > 0 && (
          <>
            <p>Строк: <b>{csvPreview.rowCount}</b>
              {csvPreview.riskPct != null && <> | Доля risk_flag=1: <b>{csvPreview.riskPct.toFixed(1)}%</b></>}
            </p>
            <div className="table-wrap">
              <table>
                <thead><tr>{csvPreview.headers.map((h) => <th key={h}>{h}</th>)}</tr></thead>
                <tbody>{csvPreview.rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={`${i}_${j}`}>{c}</td>)}</tr>)}</tbody>
              </table>
            </div>
          </>
        )}
      </div>
      <div className="card">
        <h2>2) Корреляционный анализ</h2>
        <button onClick={onCorrelation} disabled={!canRun}>Запустить корреляцию</button>
        {corrResult ? (
          <>
            <p>Размер: <b>{corrResult.n_rows}</b> строк × <b>{corrResult.n_columns}</b> колонок</p>
            {renderCorrelationTable()}
            <Plot
              data={[{
                z: cols.map((r) => cols.map((c) => Number(matrix[r]?.[c] ?? 0))),
                x: cols,
                y: cols,
                type: "heatmap",
                colorscale: "RdBu",
                zmid: 0,
              }]}
              layout={{ title: "Корреляционная матрица", autosize: true, height: 500 }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </>
        ) : <p className="muted">После запуска появится корреляционная матрица.</p>}
      </div>
      <div className="card">
        <h2>3) Асинхронные ML-задачи (RQ)</h2>
        <div className="row">
          <button onClick={onTrain} disabled={!canRun}>Обучение (RQ)</button>
          <button onClick={onShap} disabled={!canRun}>SHAP (RQ)</button>
        </div>
        {error && <p className="error">{error}</p>}
      </div>
      <PollingTask taskId={trainTaskId} title="Обучение модели" />
      <PollingTask taskId={shapTaskId} title="SHAP объяснения" />
      {trainTaskId && (
        <div className="card">
          <h2>📊 Метрики модели</h2>
          <p className="muted">После завершения обучения здесь появится график метрик</p>
          <Plot
            data={[{
              x: ["F1", "ROC-AUC", "Precision", "Recall"],
              y: [0.82, 0.88, 0.80, 0.84],
              type: "bar",
              marker: { color: "#2b6fff" },
            }]}
            layout={{ title: "Метрики модели (демо)", yaxis: { range: [0, 1] }, height: 350 }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </div>
      )}
    </>
  );
}

export default function App() {
  const [health, setHealth] = useState("checking");

  useEffect(() => {
    healthcheck().then(() => setHealth("ok")).catch(() => setHealth("down"));
  }, []);

  return (
    <HashRouter>
      <main className="container">
        <h1>АРМ исследователя — Мониторинг академических рисков</h1>
        <p className="muted">React MFE · FastAPI API · ml_core</p>
        <p>API: <span className={health === "ok" ? "ok" : health === "down" ? "error" : ""}>{health}</span></p>
        <Tabs />
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/imputation" element={<Imputation />} />
          <Route path="/crosstab" element={<Crosstab />} />
          <Route path="/timeseries" element={<TimeSeries />} />
          <Route path="/composite" element={<CompositeScore />} />
          <Route path="/drift" element={<DriftCheck />} />
          <Route path="/experiments" element={<Experiments />} />
        </Routes>
      </main>
    </HashRouter>
  );
}
