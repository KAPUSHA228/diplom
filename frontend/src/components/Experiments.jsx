import { useState } from "react";
import { listExperiments, getExperiment, saveExperiment } from "../api";

const LAST_ANALYSIS_KEY = "last_analysis_result";

export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // Сохранение текущего анализа
  const [saveName, setSaveName] = useState("");
  const [saveDesc, setSaveDesc] = useState("");
  const [saved, setSaved] = useState(false);

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

  async function onSaveExperiment() {
    if (!saveName.trim()) { setError("Введите имя эксперимента"); return; }

    // Берём последний анализ из localStorage
    const raw = localStorage.getItem(LAST_ANALYSIS_KEY);
    if (!raw) { setError("Нет результатов анализа для сохранения. Сначала запустите анализ на главной."); return; }

    setBusy(true); setError(""); setSaved(false);
    try {
      const analysis = JSON.parse(raw);
      const res = await saveExperiment(
        saveName.trim(),
        analysis.test_metrics || analysis.metrics || {},
        analysis.selected_features || [],
        saveDesc.trim()
      );
      setSaved(true);
      setSaveName("");
      setSaveDesc("");
      // Обновим список
      const listRes = await listExperiments();
      setExperiments(listRes.experiments || []);
    } catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  return (
    <div className="card">
      <h2>📁 История экспериментов</h2>

      {/* --- Сохранить текущий анализ --- */}
      <div className="save-experiment-section" style={{ marginTop: 12, padding: 12, background: "var(--bg-secondary)", borderRadius: 8 }}>
        <h3>💾 Сохранить последний анализ</h3>
        <label style={{ display: "block", marginBottom: 8 }}>
          <b>Название:</b>
          <input
            type="text"
            value={saveName}
            onChange={e => setSaveName(e.target.value)}
            placeholder="Например: Вильямс + Шварц, RF модель"
            style={{ width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)" }}
          />
        </label>
        <label style={{ display: "block", marginBottom: 8 }}>
          <b>Описание (необязательно):</b>
          <textarea
            value={saveDesc}
            onChange={e => setSaveDesc(e.target.value)}
            placeholder="Комментарий к эксперименту..."
            rows={2}
            style={{ width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)", resize: "vertical" }}
          />
        </label>
        <button className="primary" onClick={onSaveExperiment} disabled={busy || !saveName.trim()}>
          {busy ? "⏳ Сохранение..." : "✅ Сохранить эксперимент"}
        </button>
        {saved && <p className="ok" style={{ marginTop: 8 }}>✅ Эксперимент сохранён!</p>}
      </div>

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
          <details open style={{ marginTop: 8 }}>
            <summary>Детали эксперимента</summary>
            <pre>{JSON.stringify(detail, null, 2)}</pre>
          </details>
        )}
      </div>
    </div>
  );
}
