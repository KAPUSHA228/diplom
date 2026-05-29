import { memo, useState, useRef, useEffect } from "react";
import Plot from "react-plotly.js";
import { savePlotToFile } from "../api";
import downloadJSONAsCSV from "./utils/csvHelpers";

const PlotChart = memo(({ data, title, height = 400 }) => {
  const [isSaving, setIsSaving] = useState(false);
  const [format, setFormat] = useState("png");
  const plotRef = useRef(null);
  const [mode, setMode] = useState("zoom");

  const handleSavePlot = async () => {
    if (!data || !data.data) return;
    setIsSaving(true);
    try {
      await savePlotToFile(data, title.replace(/[^a-zа-я0-9]/gi, "_"), format);
    } catch (err) {
      console.error("Failed to save plot:", err);
    } finally {
      setIsSaving(false);
    }
  };

  const getPlotlyFigure = () => {
    if (plotRef.current && plotRef.current.el) {
      const plotDiv = plotRef.current.el;
      // Проверяем, инициализирован ли Plotly в этом div
      if (plotDiv && plotDiv._fullLayout) {
        return plotDiv;
      }
    }
    return null;
  };

  const handleZoomIn = () => {
    const fig = getPlotlyFigure();
    if (fig && window.Plotly) {
      const xRange = fig.layout?.xaxis?.range;
      const yRange = fig.layout?.yaxis?.range;

      if (xRange && yRange) {
        const xCenter = (xRange[0] + xRange[1]) / 2;
        const xHalf = (xRange[1] - xRange[0]) / 2;
        const yCenter = (yRange[0] + yRange[1]) / 2;
        const yHalf = (yRange[1] - yRange[0]) / 2;

        // Уменьшаем область на 30% (приближаем)
        window.Plotly.relayout(fig, {
          "xaxis.range": [xCenter - xHalf * 0.7, xCenter + xHalf * 0.7],
          "yaxis.range": [yCenter - yHalf * 0.7, yCenter + yHalf * 0.7],
        });
      } else {
        window.Plotly.relayout(fig, {
          "xaxis.autorange": true,
          "yaxis.autorange": true,
        });
        setTimeout(() => handleZoomIn(), 50);
      }
    }
  };

  const handleZoomOut = () => {
    const fig = getPlotlyFigure();
    if (fig && window.Plotly) {
      const xRange = fig.layout?.xaxis?.range;
      const yRange = fig.layout?.yaxis?.range;

      if (xRange && yRange) {
        const xCenter = (xRange[0] + xRange[1]) / 2;
        const xHalf = (xRange[1] - xRange[0]) / 2;
        const yCenter = (yRange[0] + yRange[1]) / 2;
        const yHalf = (yRange[1] - yRange[0]) / 2;

        // Увеличиваем область на 30% (отдаляем)
        window.Plotly.relayout(fig, {
          "xaxis.range": [xCenter - xHalf / 0.7, xCenter + xHalf / 0.7],
          "yaxis.range": [yCenter - yHalf / 0.7, yCenter + yHalf / 0.7],
        });
      } else {
        window.Plotly.relayout(fig, {
          "xaxis.autorange": true,
          "yaxis.autorange": true,
        });
      }
    }
  };

  const handleAutoScale = () => {
    const fig = getPlotlyFigure();
    if (fig && window.Plotly) {
      window.Plotly.relayout(fig, {
        "xaxis.autorange": true,
        "yaxis.autorange": true,
      });
    }
  };

  const handleResetAxes = () => {
    const fig = getPlotlyFigure();
    if (fig) {
      window.Plotly.relayout(fig, {
        "xaxis.autorange": true,
        "yaxis.autorange": true,
      });
    }
  };

  const handleSetMode = (newMode) => {
    setMode(newMode);
    const fig = getPlotlyFigure();
    if (fig) {
      window.Plotly.update(
        fig,
        {},
        { dragmode: newMode === "zoom" ? "zoom" : "pan" },
      );
    }
  };

  if (!data || !data.data) return <p className="muted">График не доступен</p>;

  return (
    <div style={{ width: "100%", height, position: "relative" }}>
      {/* Панель управления */}
      <div
        style={{
          position: "absolute",
          top: 8,
          right: 8,
          zIndex: 10,
          display: "flex",
          gap: 8,
          background: "var(--bg-secondary)",
          padding: "6px 12px",
          borderRadius: 8,
          border: "1px solid var(--border)",
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", gap: 4 }}>
          <button
            onClick={() => handleSetMode("zoom")}
            style={{
              padding: "4px 8px",
              fontSize: 12,
              background:
                mode === "zoom" ? "var(--primary)" : "var(--bg-secondary)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              cursor: "pointer",
              color: mode === "zoom" ? "white" : "var(--text)",
            }}
          >
            🔍 Zoom
          </button>
          <button
            onClick={() => handleSetMode("pan")}
            style={{
              padding: "4px 8px",
              fontSize: 12,
              background:
                mode === "pan" ? "var(--primary)" : "var(--bg-secondary)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              cursor: "pointer",
              color: mode === "pan" ? "white" : "var(--text)",
            }}
          >
            ✋ Pan
          </button>
        </div>

        <div
          style={{ width: 1, background: "var(--border)", margin: "0 4px" }}
        />

        <button
          onClick={handleZoomIn}
          style={{
            padding: "4px 8px",
            fontSize: 12,
            background: "var(--bg-secondary)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            color: "var(--text)",
            cursor: "pointer",
          }}
        >
          🔍+
        </button>
        <button
          onClick={handleZoomOut}
          style={{
            padding: "4px 8px",
            fontSize: 12,
            background: "var(--bg-secondary)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            color: "var(--text)",
            cursor: "pointer",
          }}
        >
          🔍-
        </button>
        <button
          onClick={handleAutoScale}
          style={{
            padding: "4px 8px",
            fontSize: 12,
            background: "var(--bg-secondary)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            color: "var(--text)",
            cursor: "pointer",
          }}
        >
          📐 Auto
        </button>
        <button
          onClick={handleResetAxes}
          style={{
            padding: "4px 8px",
            fontSize: 12,
            background: "var(--bg-secondary)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            color: "var(--text)",
            cursor: "pointer",
          }}
        >
          🔄 Reset
        </button>

        <div
          style={{ width: 1, background: "var(--border)", margin: "0 4px" }}
        />

        <select
          value={format}
          onChange={(e) => setFormat(e.target.value)}
          style={{
            padding: "4px 8px",
            fontSize: 12,
            background: "var(--bg-secondary)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            cursor: "pointer",
            color: "var(--text)",
          }}
        >
          <option value="png">PNG</option>
          <option value="svg">SVG</option>
          <option value="pdf">PDF</option>
        </select>
        <button
          onClick={handleSavePlot}
          disabled={isSaving}
          style={{
            padding: "4px 8px",
            fontSize: 12,
            background: "var(--bg-secondary)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            color: "var(--text)",
            cursor: "pointer",
          }}
        >
          {isSaving ? "⏳" : "💾 Скачать"}
        </button>
      </div>

      {/* Plotly график */}
      <Plot
        ref={plotRef}
        data={data.data}
        layout={{
          ...data.layout,
          title,
          height,
          autosize: true,
          dragmode: mode === "zoom" ? "zoom" : "pan",
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        config={{
          responsive: true,
          displayModeBar: false,
          displaylogo: false,
          scrollZoom: true,
          doubleClick: "reset+autosize",
        }}
      />
    </div>
  );
});

PlotChart.displayName = "PlotChart";

/**
 * Отображает результаты полного анализа
 */
const AnalysisResults = memo(({ result }) => {
  console.log("=== ВСЕ ДАННЫЕ result ===", result);
  console.log("=== target_col в result ===", result.target_col);
  console.log("=== КЛЮЧИ result ===", Object.keys(result));
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
  const renderStartTime = useRef(Date.now());

  // === ОТЛАДКА ===
  console.log("🔍 fig_corr в AnalysisResults:", fig_corr);
  console.log("🔍 fig_corr data:", fig_corr?.data);
  console.log("=== Данные графиков от сервера ===");
  console.log("ROC:", fig_roc);
  console.log("CM:", fig_cm);
  console.log("FI:", fig_fi);
  console.log("Clusters:", fig_clusters);
  useEffect(() => {
    console.log("🔵 НАЧАЛО РЕНДЕРА AnalysisResults");
    console.log(
      "🔵 Время с момента получения:",
      (Date.now() - renderStartTime.current) / 1000,
      "сек",
    );
  }, []);
  if (!result) return null;

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

  return (
    <div className="analysis-results">
      {/* Метрики на тесте */}
      <section className="card">
        <h2>Метрики на тестовой выборке</h2>
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
          <h2>Кросс-валидация</h2>
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
                  <td>
                    <b>{name}</b>
                  </td>
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
          <h2>Отобранные признаки ({selected_features.length})</h2>
          <div className="tags">
            {selected_features.map((f) => (
              <span key={f} className="tag">
                {f}
              </span>
            ))}
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
          <h2>Матрица Ошибок</h2>
          <PlotChart data={fig_cm} title="Confusion Matrix" />
        </section>

        <section className="card">
          <h2>Важность признаков</h2>
          <PlotChart data={fig_fi} title="Feature Importance" />
        </section>

        <section className="card">
          <h2>Корреляционная матрица</h2>
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
          <h2>Профили кластеров</h2>
          <ClusterTable profiles={cluster_profiles} />
        </section>
      )}

      {/* SHAP объяснения */}
      {explanations && explanations.length > 0 && (
        <section className="card">
          <h2>
            SHAP объяснения (топ студентов: {result.target_col || "Target"})
          </h2>
          {explanations.slice(0, 5).map((exp, i) => (
            <details key={i} className="shap-exp">
              <summary>
                Студент #{exp.student_index ?? i} — Вероятность:{" "}
                {((exp.risk_probability || 0) * 100).toFixed(1)}%
              </summary>
              <pre>{exp.explanation}</pre>
            </details>
          ))}
        </section>
      )}

      {/* === ПАНЕЛЬ ЭКСПОРТА === */}
      <div
        className="card"
        style={{
          display: "flex",
          gap: 10,
          alignItems: "center",
          marginTop: 16,
          flexWrap: "wrap",
        }}
      >
        <h3 style={{ margin: 0, marginRight: "auto" }}> Экспорт</h3>
        <button
          onClick={handleExportPredictions}
          disabled={!result.predictions?.length}
        >
          Предсказания (CSV)
        </button>
        <button
          onClick={handleExportExplanations}
          disabled={!explanations.length}
        >
          SHAP Объяснения (CSV)
        </button>
        <button
          onClick={handleExportClusters}
          disabled={!Object.keys(cluster_profiles).length}
        >
          Профили кластеров (CSV)
        </button>
      </div>
    </div>
  );
});
AnalysisResults.displayName = "AnalysisResults";

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {typeof value === "number" ? value.toFixed(4) : "—"}
      </div>
    </div>
  );
}

function ClusterTable({ profiles }) {
  const rows = Array.isArray(profiles)
    ? profiles
    : Object.entries(profiles).map(([k, v]) => ({ cluster: k, ...v }));
  if (rows.length === 0) return <p className="muted">Нет данных о кластерах</p>;

  const allKeys = new Set();
  rows.forEach((r) => Object.keys(r).forEach((k) => allKeys.add(k)));
  const cols = [...allKeys];

  return (
    <div className="table-wrap">
      <table className="matrix">
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c}>
                  {typeof r[c] === "number" ? r[c].toFixed(2) : r[c]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default AnalysisResults;
