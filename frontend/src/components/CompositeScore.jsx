import { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import { createCompositeScore } from "../api";
import { parseFile } from "../utils/parseFile";
import { useSharedData } from "../hooks/useSharedData";

export default function CompositeScore() {
  const { data: sharedData, columns: sharedCols, hasShared, updateData } = useSharedData();
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [weights, setWeights] = useState({});
  const [scoreName, setScoreName] = useState("custom_score");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const activeData = fileData || sharedData;
  const numericCols = activeData?.length ? Object.keys(activeData[0]).filter(k => typeof activeData[0][k] === "number") : [];

  useEffect(() => {
    // Инициализируем веса нулями
    if (hasShared && !fileData && Object.keys(weights).length === 0) {
      const w = {};
      numericCols.forEach(c => { w[c] = 0; });
      setWeights(w);
    }
  }, [hasShared, fileData, numericCols]);

  async function onFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setResult(null);
    try {
      const parsed = await parseFile(f);
      setFileData(parsed?.allData || null);
      if (parsed?.allData?.length) {
        const w = {};
        Object.keys(parsed.allData[0]).filter(k => typeof parsed.allData[0][k] === "number").forEach(c => { w[c] = 1; });
        setWeights(w);
      }
    } catch { /* ignore */ }
  }

  async function onRun() {
    if (!activeData || Object.keys(weights).length === 0) return;
    setBusy(true); setError(""); setResult(null);
    try {
      const cleanWeights = {};
      Object.entries(weights).forEach(([k, v]) => { cleanWeights[k] = Number(v) || 0; });
      const res = await createCompositeScore(activeData, cleanWeights, scoreName);
      setResult(res);
    } catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  // Логика сохранения нового признака в общий датасет
  async function onSaveToDataset() {
    if (!result || !result.values || !activeData) return;

    try {
      // Создаем новый массив данных с добавленной колонкой
      const newData = activeData.map((row, idx) => ({
        ...row,
        [result.score_name]: result.values[idx] || 0
      }));

      // Обновляем общий датасет (это обновит и sessionStorage)
      updateData(newData);

      // Если данные были загружены из файла локально, обновляем и их
      if (fileData) {
        setFileData(newData);
      }

      alert(`✅ Признак "${result.score_name}" добавлен в общий набор данных!`);
    } catch (e) {
      setError("Ошибка сохранения: " + e.message);
    }
  }

  return (
    <div className="card">
      <h2>🎯 Конструктор композитных оценок</h2>
      <p className="muted">Создайте интегральный показатель, взвешивая важные признаки.</p>

      <div style={{ marginBottom: 12 }}>
        {hasShared && !fileData && (
          <div className="ok" style={{ padding: 8, borderRadius: 6, background: "var(--bg-secondary)", marginBottom: 8 }}>
            ✅ Данные с главной: <b>{sharedData.length} строк</b>, {numericCols.length} числовых колонок
          </div>
        )}
        <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileChange} />
      </div>

      {numericCols.length > 0 && (
        <>
          <label><b>Название оценки:</b>
            <input type="text" value={scoreName} onChange={e => setScoreName(e.target.value)}
              style={{ marginLeft: 8, padding: 4, borderRadius: 4, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)" }} />
          </label>

          <div style={{ marginTop: 12 }}>
            <label><b>Веса признаков:</b> <span className="muted" style={{ fontSize: 12 }}>(можно вводить дробные, например 0.5 или -1.2)</span></label>
            <div style={{ maxHeight: 300, overflowY: "auto", marginTop: 8 }}>
              {numericCols.map(col => (
                <div key={col} style={{ marginBottom: 6, display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ minWidth: 140, fontSize: 13 }}>{col}</span>
                  <input type="number" step="0.1" value={weights[col] || 0}
                    onChange={e => setWeights(p => ({ ...p, [col]: parseFloat(e.target.value) || 0 }))}
                    style={{ flex: 1, padding: 4, borderRadius: 4, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)" }} />
                </div>
              ))}
            </div>
          </div>

          <button className="primary" onClick={onRun} disabled={busy} style={{ marginTop: 12 }}>
            {busy ? "⏳ Расчет..." : "🚀 Рассчитать оценку"}
          </button>
        </>
      )}

      {error && <p className="error">{error}</p>}

      {/* === РЕЗУЛЬТАТЫ === */}
      {result && (
        <div style={{ marginTop: 20, borderTop: "1px solid var(--border)", paddingTop: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <h3 style={{ margin: 0 }}>📊 Результаты: {result.score_name}</h3>
            <button onClick={onSaveToDataset} style={{ padding: "6px 12px", borderRadius: 6, border: "none", background: "var(--primary)", color: "#fff", cursor: "pointer" }}>
              💾 Добавить в набор данных
            </button>
          </div>

          {/* Статистика */}
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 20 }}>
            {result.statistics && Object.entries(result.statistics).map(([k, v]) => (
              <div key={k} style={{ background: "var(--bg-secondary)", padding: 10, borderRadius: 6, minWidth: 100 }}>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{k}</div>
                <div style={{ fontSize: 18, fontWeight: "bold" }}>{typeof v === "number" ? v.toFixed(2) : v}</div>
              </div>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            {/* Гистограмма распределения */}
            <div>
              <h4>Распределение оценки</h4>
              {result.values && (
                <Plot
                  data={[{ x: result.values, type: "histogram", marker: { color: "var(--primary)", opacity: 0.8 } }]}
                  layout={{ margin: { l: 40, r: 20, t: 20, b: 40 }, height: 300, xaxis: { title: "Значение" } }}
                  config={{ displayModeBar: false }}
                  style={{ width: "100%" }}
                />
              )}
            </div>

            {/* Топ корреляций */}
            <div>
              <h4>Топ связей с признаками</h4>
              {result.correlations && Object.keys(result.correlations).length > 0 ? (
                <Plot
                  data={[{
                    x: Object.values(result.correlations).slice(0, 5),
                    y: Object.keys(result.correlations).slice(0, 5),
                    type: "bar",
                    orientation: "h",
                    text: Object.values(result.correlations).slice(0, 5).map(v => v.toFixed(2)),
                    textposition: "outside",
                    textfont: { size: 14, color: "#fff" },
                    marker: {
                      color: Object.values(result.correlations).slice(0, 5).map(v => v > 0 ? "#2ecc71" : "#e74c3c"),
                      opacity: 0.9
                    }
                  }]}
                  layout={{
                    margin: { l: 120, r: 40, t: 20, b: 40 },
                    height: 300,
                    xaxis: { title: "Коэффициент корреляции (r)" },
                    bargap: 0.3
                  }}
                  config={{ displayModeBar: false }}
                  style={{ width: "100%" }}
                />
              ) : <p className="muted">Нет значимых корреляций</p>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
