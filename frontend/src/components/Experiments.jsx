import { useState } from "react";
import { listExperiments, getExperiment } from "../api";

export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onLoadList() {
    setBusy(true); setError("");
    try {
      const res = await listExperiments();
      setExperiments(res.experiments || []);
    } catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  async function onLoadDetail() {
    if (!selected) return;
    setBusy(true); setError("");
    try {
      const data = await getExperiment(selected);
      setDetail(data);
    } catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  return (
    <div className="card">
      <h2>История экспериментов</h2>
      <button onClick={onLoadList} disabled={busy}>Загрузить список</button>
      {error && <p className="error">{error}</p>}
      {experiments.length > 0 && (
        <>
          <div className="table-wrap" style={{ marginTop: 8 }}>
            <table>
              <thead><tr><th>ID</th><th>Имя</th><th>Модель</th><th>Признаков</th></tr></thead>
              <tbody>
                {experiments.map((e, i) => (
                  <tr key={i}>
                    <td>{e.id}</td>
                    <td>{e.name}</td>
                    <td>{e.model_name || "—"}</td>
                    <td>{e.n_features || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            <select value={selected} onChange={(e) => setSelected(e.target.value)}>
              <option value="">— Выберите эксперимент —</option>
              {experiments.map((e) => <option key={e.id} value={e.id}>{e.name} ({e.id})</option>)}
            </select>
            <button onClick={onLoadDetail} disabled={!selected || busy}>Загрузить</button>
          </div>
        </>
      )}
      {detail && (
        <details open style={{ marginTop: 8 }}>
          <summary>Детали эксперимента</summary>
          <pre>{JSON.stringify(detail, null, 2)}</pre>
        </details>
      )}
    </div>
  );
}
