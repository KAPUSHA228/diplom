import { useState } from "react";
import { listExperiments, getExperiment } from "../api";

const LAST_ANALYSIS_KEY = "last_analysis_result";

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
      <h2>📁 История экспериментов</h2>
      <p className="muted">Чтобы сохранить анализ, используйте форму на вкладке "Главное" после завершения анализа.</p>

      {/* --- Список экспериментов --- */}
      <div style={{ marginTop: 16 }}>
        <h3>Список сохранённых экспериментов</h3>
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
          <div style={{ marginTop: 16, padding: 12, background: "var(--bg-secondary)", borderRadius: 8 }}>
            <h3 style={{ margin: "0 0 12px 0" }}>📄 Детали: {detail.name}</h3>

            {/* Таблица конфигурации */}
            {detail.config && Object.keys(detail.config).length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <h4 style={{ margin: "0 0 6px 0" }}>⚙️ Параметры:</h4>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  {Object.entries(detail.config).map(([key, val]) => (
                    <div key={key} style={{ display: "flex", justifyContent: "space-between", padding: "4px 8px", background: "var(--bg)", borderRadius: 4 }}>
                      <span style={{ fontWeight: 600 }}>{key}:</span>
                       <span>{String(val)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Метрики */}
            <div style={{ marginBottom: 12 }}>
              <h4 style={{ margin: "0 0 6px 0" }}>📊 Метрики:</h4>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8 }}>
                {detail.metrics && Object.entries(detail.metrics).map(([k, v]) => (
                  <div key={k} style={{ textAlign: "center", padding: "4px", background: "var(--bg)", borderRadius: 4 }}>
                    <div style={{ fontSize: 10 }}>{k}</div>
                    <div style={{ fontWeight: "bold" }}>{typeof v === "number" ? v.toFixed(3) : v}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Признаки */}
            {detail.features && detail.features.length > 0 && (
              <div>
                <h4 style={{ margin: "0 0 6px 0" }}>🧬 Признаки:</h4>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {detail.features.map((f, i) => (
                    <span key={i} style={{ padding: "2px 6px", background: "var(--primary)", color: "#fff", borderRadius: 4, fontSize: 12 }}>{f}</span>
                  ))}
                </div>
              </div>
            )}

            {detail.description && <p style={{ marginTop: 12, fontStyle: "italic" }}>{detail.description}</p>}
          </div>
        )}
      </div>
    </div>
  );
}
