import { useState } from "react";
import { getTrajectory, findNegativeDynamics, forecastStudent } from "../api";

export default function TimeSeries() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [valueCol, setValueCol] = useState("");
  const [timeCol, setTimeCol] = useState("semester");
  const [result, setResult] = useState(null);
  const [dynamics, setDynamics] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [selectedStudent, setSelectedStudent] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onFileChange(e) {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setResult(null); setDynamics(null); setForecast(null);
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
      if (headers.includes("avg_grade")) setValueCol("avg_grade");
      if (headers.length > 2 && !valueCol) setValueCol(headers[2]);
    } catch { /* ignore */ }
  }

  async function onTrajectory() {
    if (!data || !valueCol || !selectedStudent) return;
    setBusy(true); setError("");
    try {
      const res = await getTrajectory(data, selectedStudent, valueCol, timeCol);
      setResult(res);
    } catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  async function onNegativeDynamics() {
    if (!data || !valueCol) return;
    setBusy(true); setError("");
    try {
      const res = await findNegativeDynamics(data, valueCol, timeCol);
      setDynamics(res);
    } catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  async function onForecast() {
    if (!data || !valueCol || !selectedStudent) return;
    setBusy(true); setError("");
    try {
      const res = await forecastStudent(data, selectedStudent, valueCol, timeCol);
      setForecast(res);
    } catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  const students = data ? [...new Set(data.map((d) => d.student_id ?? d.StudentID ?? d.user_id))] : [];

  return (
    <div className="card">
      <h2>Временные ряды и траектории</h2>
      <input type="file" accept=".csv" onChange={onFileChange} />
      {data && (
        <>
          <div className="row" style={{ marginTop: 8 }}>
            <label>Показатель: </label>
            <select value={valueCol} onChange={(e) => setValueCol(e.target.value)}>
              {Object.keys(data[0]).filter((k) => typeof data[0][k] === "number").map((h) => <option key={h}>{h}</option>)}
            </select>
            <label>Студент: </label>
            <select value={selectedStudent} onChange={(e) => setSelectedStudent(e.target.value)}>
              <option value="">—</option>
              {students.map((s) => <option key={s} value={s}>{s}</option>)}
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
              <p>Тренд: <b>{result.trend?.toFixed(3)}</b> | Статус: <b>{result.status}</b></p>
            </div>
          )}
          {dynamics && (
            <div>
              <p>Проанализировано: <b>{dynamics.n_students_analyzed}</b> | В зоне риска: <b>{dynamics.at_risk_count}</b> ({dynamics.risk_percentage}%)</p>
              {dynamics.at_risk_students?.length > 0 && (
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Студент</th><th>Тренд</th><th>Начало</th><th>Конец</th></tr></thead>
                    <tbody>
                      {dynamics.at_risk_students.slice(0, 10).map((s, i) => (
                        <tr key={i}><td>{s.student_id}</td><td>{s.trend?.toFixed(3)}</td><td>{s.first_value}</td><td>{s.last_value}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
          {forecast && (
            <div>
              <p><b>Прогноз:</b></p>
              {forecast.future_semesters?.map((s, i) => (
                <p key={i}>Семестр {s}: <b>{forecast.predictions[i]?.toFixed(2)}</b></p>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
