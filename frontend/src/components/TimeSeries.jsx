import { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import { getTrajectory, findNegativeDynamics, forecastStudent } from "../api";
import { parseFile } from "../utils/parseFile";
import { useSharedData } from "../hooks/useSharedData";

export default function TimeSeries() {
  const { data: sharedData, columns: sharedCols, hasShared } = useSharedData();
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [valueCol, setValueCol] = useState("");
  const [timeCol, setTimeCol] = useState("semester");
  const [result, setResult] = useState(null);
  const [dynamics, setDynamics] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [selectedStudent, setSelectedStudent] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const activeData = fileData || sharedData;
  const allStudents = activeData ? [...new Set(activeData.map(d => d.student_id ?? d.StudentID ?? d.user_id).filter(Boolean))] : [];

  function initCols(data) {
    if (!data?.length) return;
    const keys = Object.keys(data[0]);
    if (keys.includes("avg_grade")) setValueCol("avg_grade");
    else { const num = keys.find(k => typeof data[0][k] === "number"); if (num) setValueCol(num); }
  }

  useEffect(() => {
    if (hasShared && !fileData && !valueCol) initCols(sharedData);
  }, [hasShared, fileData]);

  async function onFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setResult(null); setDynamics(null); setForecast(null);
    try {
      const parsed = await parseFile(f);
      setFileData(parsed?.allData || null);
      initCols(parsed?.allData);
    } catch { /* ignore */ }
  }

  async function onTrajectory() {
    if (!activeData || !valueCol || !selectedStudent) return;
    setBusy(true); setError("");
    try { setResult(await getTrajectory(activeData, selectedStudent, valueCol, timeCol)); }
    catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  async function onNegativeDynamics() {
    if (!activeData || !valueCol) return;
    setBusy(true); setError("");
    try { setDynamics(await findNegativeDynamics(activeData, valueCol, timeCol)); }
    catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  async function onForecast() {
    if (!activeData || !valueCol || !selectedStudent) return;
    setBusy(true); setError("");
    try { setForecast(await forecastStudent(activeData, selectedStudent, valueCol, timeCol)); }
    catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  return (
    <div className="card">
      <h2>📉 Временные ряды и траектории</h2>

      <div style={{ marginBottom: 12 }}>
        {hasShared && !fileData && (
          <div className="ok" style={{ padding: 8, borderRadius: 6, background: "var(--bg-secondary)", marginBottom: 8 }}>
            ✅ Используются данные с главной: <b>{sharedData.length} строк</b>, {sharedCols.length} колонок
          </div>
        )}
        <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileChange} />
      </div>

      {activeData && activeData.length > 0 && (
        <>
          <div className="row" style={{ marginTop: 8 }}>
            <label>Показатель: </label>
            <select value={valueCol} onChange={(e) => setValueCol(e.target.value)}>
              {Object.keys(activeData[0]).filter((k) => typeof activeData[0][k] === "number").map((h) => <option key={h}>{h}</option>)}
            </select>
            <label>Студент: </label>
            <select value={selectedStudent} onChange={(e) => setSelectedStudent(e.target.value)}>
              <option value="">—</option>
              {allStudents.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            <button onClick={onTrajectory} disabled={busy || !selectedStudent}>Траектория</button>
            <button onClick={onNegativeDynamics} disabled={busy}>Негативная динамика</button>
            <button onClick={onForecast} disabled={busy || !selectedStudent}>Прогноз</button>
          </div>
          {error && <p className="error">{error}</p>}

          {result && (
            <div>
              <h3>📈 Траектория студента {selectedStudent}</h3>
              <p>Тренд: <b>{result.trend?.toFixed(3)}</b> | Статус: <b>{result.status}</b></p>
              {result.chart && <Plot data={result.chart.data} layout={{ ...result.chart.layout, height: 350 }} config={{ responsive: true, displayModeBar: false }} style={{ width: "100%" }} />}
            </div>
          )}
          {dynamics && (
            <div>
              <p>Проанализировано: <b>{dynamics.n_students_analyzed}</b> | В зоне риска: <b>{dynamics.at_risk_count}</b> ({dynamics.risk_percentage?.toFixed(1)}%)</p>
              {dynamics.at_risk_students?.length > 0 && (
                <div className="table-wrap"><table><thead><tr><th>Студент</th><th>Тренд</th><th>Начало</th><th>Конец</th></tr></thead><tbody>{dynamics.at_risk_students.slice(0, 10).map((s, i) => (<tr key={i}><td>{s.student_id}</td><td>{s.trend?.toFixed(3)}</td><td>{s.first_value}</td><td>{s.last_value}</td></tr>))}</tbody></table></div>
              )}
            </div>
          )}
          {forecast && (
            <div>
              <h3>🔮 Прогноз</h3>
              {forecast.chart && <Plot data={forecast.chart.data} layout={{ ...forecast.chart.layout, height: 350 }} config={{ responsive: true, displayModeBar: false }} style={{ width: "100%" }} />}
              {!forecast.chart && forecast.future_semesters?.map((s, i) => (<p key={i}>Семестр {s}: <b>{forecast.predictions[i]?.toFixed(2)}</b></p>))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
