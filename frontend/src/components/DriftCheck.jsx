import { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import { checkDrift, getMetricsHistory } from "../api";
import { parseFile } from "../utils/parseFile";
import { useSharedData } from "../hooks/useSharedData";

export default function DriftCheck() {
  const { data: sharedData, hasShared } = useSharedData();
  const [refFile, setRefFile] = useState(null);
  const [refData, setRefData] = useState(null);
  const [curFile, setCurFile] = useState(null);
  const [curData, setCurData] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // История метрик
  const [metrics, setMetrics] = useState([]);

  useEffect(() => {
    getMetricsHistory().then(r => setMetrics(r.metrics || [])).catch(() => {});
  }, []);

  // Shared data как референс по умолчанию
  useEffect(() => {
    if (hasShared && !refData) setRefData(sharedData);
  }, [hasShared, sharedData]);

  async function onRefChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setRefFile(f); setResult(null);
    try {
      const parsed = await parseFile(f);
      setRefData(parsed?.allData || null);
    } catch { /* ignore */ }
  }

  async function onCurChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setCurFile(f); setResult(null);
    try {
      const parsed = await parseFile(f);
      setCurData(parsed?.allData || null);
    } catch { /* ignore */ }
  }

  async function onRun() {
    if (!refData || !curData) return;
    setBusy(true); setError("");
    try { setResult(await checkDrift(refData, curData)); }
    catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  return (
    <div className="card">
      <h2>🔄 Проверка дрейфа данных</h2>

      <div style={{ marginBottom: 12 }}>
        {hasShared && !refFile && (
          <div className="ok" style={{ padding: 8, borderRadius: 6, background: "var(--bg-secondary)", marginBottom: 8 }}>
            ✅ Эталон: данные с главной (<b>{sharedData.length} строк</b>). Загрузите новые данные для сравнения.
          </div>
        )}
        <div className="row">
          <div>
            <label><b>Эталон:</b></label>
            <input type="file" accept=".csv,.xlsx,.xls" onChange={onRefChange} />
          </div>
          <div>
            <label><b>Новые данные:</b></label>
            <input type="file" accept=".csv,.xlsx,.xls" onChange={onCurChange} />
          </div>
        </div>
      </div>

      <button onClick={onRun} disabled={busy || !refData || !curData}>Проверить дрейф</button>
      {error && <p className="error">{error}</p>}

      {result && (
        <div style={{ marginTop: 12 }}>
          <p>Статус: {result.overall_drift ? <span className="error">🔴 Есть дрейф</span> : <span className="ok">🟢 Нет дрейфа</span>}</p>
          <p>Процент дрейфа: <b>{result.drift_percentage}%</b></p>
          {result.drifted_features?.length > 0 && <p>Дрейфующие признаки: {result.drifted_features.join(", ")}</p>}
          {result.recommendations?.length > 0 && <details><summary>Рекомендации</summary>{result.recommendations.map((r, i) => <p key={i}>• {r}</p>)}</details>}
        </div>
      )}

      {/* История метрик */}
      <div style={{ marginTop: 24 }}>
        <h3>📈 История метрик моделей</h3>
        {metrics.length > 0 ? (
          <Plot
            data={[
              { x: metrics.map((_, i) => i + 1), y: metrics.map(m => m.f1 ?? 0), name: "F1", mode: "lines+markers", type: "scatter" },
              { x: metrics.map((_, i) => i + 1), y: metrics.map(m => m.roc_auc ?? 0), name: "ROC-AUC", mode: "lines+markers", type: "scatter" },
              { x: metrics.map((_, i) => i + 1), y: metrics.map(m => m.precision ?? 0), name: "Precision", mode: "lines+markers", type: "scatter" },
              { x: metrics.map((_, i) => i + 1), y: metrics.map(m => m.recall ?? 0), name: "Recall", mode: "lines+markers", type: "scatter" },
            ]}
            layout={{ title: "Метрики по запускам", xaxis: { title: "Запуск" }, yaxis: { title: "Метрика", range: [0, 1] }, height: 400, showlegend: true, legend: { orientation: "h", y: -0.3 } }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: "100%" }}
          />
        ) : <p className="muted">Нет данных о метриках. Запустите анализ на главной.</p>}
      </div>
    </div>
  );
}
