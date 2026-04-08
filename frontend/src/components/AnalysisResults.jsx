import Plot from "react-plotly.js";

/** Хелпер для скачивания JSON как CSV */
function downloadJSONAsCSV(data, filename) {
  if (!data || !data.length) return;
  const keys = Object.keys(data[0]);
  const csvContent =
    "data:text/csv;charset=utf-8," +
    [
      keys.join(","),
      ...data.map((row) => keys.map((k) => JSON.stringify(row[k] || "")).join(",")),
    ].join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", filename + ".csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Отображает результаты полного анализа
 */
export default function AnalysisResults({ result }) {
  if (!result) return null;

  const {
    test_metrics = {},
    cv_results = {},
    selected_features = [],
    cluster_profiles = {},
    explanations = [],
    fig_cm,
    fig_roc,
    fig_fi,
    fig_clusters,
    fig_corr,
  } = result;

  // === ОТЛАДКА ===
  console.log("=== Данные графиков от сервера ===");
  console.log("ROC:", fig_roc);
  console.log("CM:", fig_cm);
  console.log("FI:", fig_fi);
  console.log("Clusters:", fig_clusters);

  // Функции экспорта
  const handleExportExplanations = () => {
    downloadJSONAsCSV(explanations, "shap_explanations");
  };

  const handleExportPredictions = () => {
    const preds = result.predictions || [];
    downloadJSONAsCSV(preds, "predictions");
  };

  const handleExportClusters = () => {
    // Превращаем объект профилей в массив строк
    const rows = Object.entries(cluster_profiles).map(([cluster, data]) => ({
      cluster,
      ...data,
    }));
    downloadJSONAsCSV(rows, "cluster_profiles");
  };

  // Хелпер для рендеринга Plotly графика из JSON
  const PlotChart = ({ data, title, height = 400 }) => {
    if (!data || !data.data) return <p className="muted">График не доступен</p>;
    return (
      <div style={{ width: "100%", height }}>
        <Plot
          data={data.data}
          layout={{ ...data.layout, title, height, autosize: true }}
          useResizeHandler
          style={{ width: "100%", height: "100%" }}
          config={{ responsive: true }}
        />
      </div>
    );
  };

  return (
    <div className="analysis-results">

      {/* Метрики на тесте */}
      <section className="card">
        <h2>📊 Метрики на тестовой выборке</h2>
        <div className="metrics-grid">
          <MetricCard label="F1-score" value={test_metrics.f1} />
          <MetricCard label="ROC-AUC" value={test_metrics.roc_auc} />
          <MetricCard label="Precision" value={test_metrics.precision} />
          <MetricCard label="Recall" value={test_metrics.recall} />
        </div>
      </section>

      {/* Кросс-валидация */}
      {cv_results && Object.keys(cv_results).length > 0 && (
        <section className="card">
          <h2>📋 Кросс-валидация</h2>
          <table className="matrix">
            <thead>
              <tr>
                <th>Модель</th>
                <th>F1 (mean)</th>
                <th>F1 (std)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(cv_results).map(([name, v]) => (
                <tr key={name}>
                  <td><b>{name}</b></td>
                  <td>{v.mean?.toFixed(4)}</td>
                  <td>± {v.std?.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {/* Отобранные признаки */}
      {selected_features.length > 0 && (
        <section className="card">
          <h2>🔍 Отобранные признаки ({selected_features.length})</h2>
          <div className="tags">
            {selected_features.map((f) => <span key={f} className="tag">{f}</span>)}
          </div>
        </section>
      )}

      {/* === ГРАФИКИ === */}
      <div className="plots-grid">
        <section className="card">
          <h2>ROC-кривые</h2>
          <PlotChart data={fig_roc} title="Receiver Operating Characteristic" />
        </section>

        <section className="card">
          <h2>Confusion Matrix</h2>
          <PlotChart data={fig_cm} title="Confusion Matrix" />
        </section>

        <section className="card">
          <h2>Важность признаков</h2>
          <PlotChart data={fig_fi} title="Feature Importance" />
        </section>

        <section className="card">
          <h2>🔗 Корреляционная матрица</h2>
          <PlotChart data={fig_corr} title="Correlation Heatmap" />
        </section>

        <section className="card">
          <h2>Кластеризация (PCA)</h2>
          <PlotChart data={fig_clusters} title="Clusters Visualization" />
        </section>
      </div>

      {/* Кластеры */}
      {cluster_profiles && Object.keys(cluster_profiles).length > 0 && (
        <section className="card">
          <h2>🎯 Профили кластеров</h2>
          <ClusterTable profiles={cluster_profiles} />
        </section>
      )}

      {/* SHAP объяснения */}
      {explanations && explanations.length > 0 && (
        <section className="card">
          <h2>💡 SHAP объяснения (топ студентов в зоне риска)</h2>
          {explanations.slice(0, 5).map((exp, i) => (
            <details key={i} className="shap-exp">
              <summary>
                Студент #{exp.student_index ?? i} — Риск: {((exp.risk_probability || 0) * 100).toFixed(1)}%
              </summary>
              <pre>{exp.explanation}</pre>
            </details>
          ))}
        </section>
      )}

      {/* === ПАНЕЛЬ ЭКСПОРТА === */}
      <div className="card" style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 16, flexWrap: "wrap" }}>
        <h3 style={{ margin: 0, marginRight: "auto" }}>💾 Экспорт</h3>
        <button onClick={handleExportPredictions} disabled={!result.predictions?.length}>
          📥 Предсказания (CSV)
        </button>
        <button onClick={handleExportExplanations} disabled={!explanations.length}>
          📥 SHAP Объяснения (CSV)
        </button>
        <button onClick={handleExportClusters} disabled={!Object.keys(cluster_profiles).length}>
          📥 Профили кластеров (CSV)
        </button>
      </div>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{typeof value === "number" ? value.toFixed(4) : "—"}</div>
    </div>
  );
}

function ClusterTable({ profiles }) {
  const rows = Array.isArray(profiles) ? profiles : Object.entries(profiles).map(([k, v]) => ({ cluster: k, ...v }));
  if (rows.length === 0) return <p className="muted">Нет данных о кластерах</p>;

  const allKeys = new Set();
  rows.forEach((r) => Object.keys(r).forEach((k) => allKeys.add(k)));
  const cols = [...allKeys];

  return (
    <div className="table-wrap">
      <table className="matrix">
        <thead>
          <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => <td key={c}>{typeof r[c] === "number" ? r[c].toFixed(2) : r[c]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
