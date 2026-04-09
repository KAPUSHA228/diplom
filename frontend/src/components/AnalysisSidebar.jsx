import { useState } from "react";

export default function AnalysisSidebar({ onRun, busy, disabled = false }) {
  const [nClusters, setNClusters] = useState(3);
  const [riskThreshold, setRiskThreshold] = useState(0.5);
  const [corrThreshold, setCorrThreshold] = useState(0.3);
  const [useSmote, setUseSmote] = useState(true);

  // Новые настройки (как в Streamlit)
  const [useHpTuning, setUseHpTuning] = useState(false);
  const [nIterTuning, setNIterTuning] = useState(20);
  const [optimizationMetric, setOptimizationMetric] = useState("default");
  const [nFeatures, setNFeatures] = useState(7);
  const [useLR, setUseLR] = useState(true);
  const [useRF, setUseRF] = useState(true);
  const [useXGB, setUseXGB] = useState(true);
  const [shapTopN, setShapTopN] = useState(5);

  function handleSubmit() {
    onRun({
      n_clusters: nClusters,
      risk_threshold: riskThreshold,
      corr_threshold: corrThreshold,
      use_smote: useSmote,
      use_hp_tuning: useHpTuning,
      n_iter_tuning: nIterTuning,
      optimization_metric: optimizationMetric,
      n_features_to_select: nFeatures,
      use_lr: useLR,
      use_rf: useRF,
      use_xgb: useXGB,
      shap_top_n: shapTopN,
    });
  }

  return (
    <aside className="sidebar">
      <h2>⚙️ Настройки анализа</h2>

      <label>
        Число кластеров: <b>{nClusters}</b>
        <input
          type="range" min="2" max="6" value={nClusters}
          onChange={(e) => setNClusters(Number(e.target.value))}
        />
      </label>

      <label>
        Порог риска: <b>{riskThreshold.toFixed(2)}</b>
        <input
          type="range" min="0" max="1" step="0.05" value={riskThreshold}
          onChange={(e) => setRiskThreshold(Number(e.target.value))}
        />
      </label>

      <label>
        Порог корреляции: <b>{corrThreshold.toFixed(2)}</b>
        <input
          type="range" min="0" max="1" step="0.05" value={corrThreshold}
          onChange={(e) => setCorrThreshold(Number(e.target.value))}
        />
      </label>

      <label>
        Признаков в финальной модели: <b>{nFeatures}</b>
        <input
          type="range" min="3" max="15" value={nFeatures}
          onChange={(e) => setNFeatures(Number(e.target.value))}
        />
      </label>

      <label>
        Кол-во факторов в SHAP: <b>{shapTopN}</b>
        <input
          type="range" min="3" max="10" value={shapTopN}
          onChange={(e) => setShapTopN(Number(e.target.value))}
        />
      </label>

      <label className="checkbox-label">
        <div style={{ marginBottom: 4, fontSize: 13, fontWeight: 600 }}>Приоритетная метрика:</div>
        <select value={optimizationMetric} onChange={(e) => setOptimizationMetric(e.target.value)}
          style={{ width: "100%", padding: 4, borderRadius: 4, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)" }}>
          <option value="default">По умолчанию (F1)</option>
          <option value="f1">F1-score</option>
          <option value="roc_auc">ROC-AUC</option>
          <option value="precision">Precision</option>
          <option value="recall">Recall</option>
        </select>
      </label>

      <div style={{ marginTop: 12, marginBottom: 8, fontWeight: 600 }}>Модели для обучения:</div>
      <label className="checkbox-label">
        <input type="checkbox" checked={useLR} onChange={(e) => setUseLR(e.target.checked)} />
        Logistic Regression
      </label>
      <label className="checkbox-label">
        <input type="checkbox" checked={useRF} onChange={(e) => setUseRF(e.target.checked)} />
        Random Forest
      </label>
      <label className="checkbox-label">
        <input type="checkbox" checked={useXGB} onChange={(e) => setUseXGB(e.target.checked)} />
        XGBoost
      </label>

      <label className="checkbox-label" style={{ marginTop: 8 }}>
        <input type="checkbox" checked={useSmote} onChange={(e) => setUseSmote(e.target.checked)} />
        SMOTE (балансировка классов)
      </label>

      <label className="checkbox-label">
        <input type="checkbox" checked={useHpTuning} onChange={(e) => setUseHpTuning(e.target.checked)} />
        Оптимизация гиперпараметров (XGB)
      </label>

      {useHpTuning && (
        <label>
          Итераций тюнинга: <b>{nIterTuning}</b>
          <input
            type="range" min="10" max="50" value={nIterTuning}
            onChange={(e) => setNIterTuning(Number(e.target.value))}
          />
        </label>
      )}

      <button className="primary" onClick={handleSubmit} disabled={busy || disabled} style={{ marginTop: 12 }}>
        {busy ? "⏳ Анализ..." : disabled ? "🎯 Сначала выберите цель" : "🚀 Запустить анализ"}
      </button>

      <p className="hint">
        {disabled
          ? "⬆️ Сначала загрузите файл и выберите целевую переменную"
          : "Настройки не влияют на данные, пока не нажата кнопка «Запустить анализ»."}
      </p>
    </aside>
  );
}
