import { useState, useEffect } from "react";
import { useDatasetLoader } from "../hooks/useDatasetLoader";

export default function AnalysisSidebar({ onRun, busy, canRun = false }) {
  const { data, loading } = useDatasetLoader();
  const [nClusters, setNClusters] = useState(3);
  const [riskThreshold, setRiskThreshold] = useState(0.5);
  const [corrThreshold, setCorrThreshold] = useState(0.3);
  const [useSmote, setUseSmote] = useState(true);
  const [useHpTuning, setUseHpTuning] = useState(false);
  const [nIterTuning, setNIterTuning] = useState(20);
  const [optimizationMetric, setOptimizationMetric] = useState("default");
  const [nFeatures, setNFeatures] = useState(7);
  const [useLR, setUseLR] = useState(true);
  const [useRF, setUseRF] = useState(true);
  const [useXGB, setUseXGB] = useState(true);
  const [shapTopN, setShapTopN] = useState(5);

  const [rowCount, setRowCount] = useState(0);
  const [isLargeDataset, setIsLargeDataset] = useState(false);
  const [isVeryLargeDataset, setIsVeryLargeDataset] = useState(false);

  useEffect(() => {
    const count = data?.length || 0;
    setRowCount(count);
    setIsLargeDataset(count > 100000);
    setIsVeryLargeDataset(count > 200000);
  }, [data]);

  const getRFWarning = () => {
    if (!useRF) return null;
    if (isVeryLargeDataset) {
      return {
        level: "error",
        message: ` Ваш файл содержит ${rowCount.toLocaleString()} строк. Random Forest может обучаться 3-5 минут! Рекомендуется отключить.`,
      };
    }
    if (isLargeDataset) {
      return {
        level: "warning",
        message: ` Ваш файл содержит ${rowCount.toLocaleString()} строк. Random Forest может обучаться 1-2 минуты.`,
      };
    }
    return null;
  };

  const rfWarning = getRFWarning();

  // Предупреждение для SMOTE на больших данных
  const getSmoteWarning = () => {
    if (!useSmote) return null;
    if (isVeryLargeDataset) {
      return {
        level: "warning",
        message: `SMOTE на ${rowCount.toLocaleString()} строках может занять ~6 секунд. При необходимости можно отключить.`,
      };
    }
    return null;
  };

  const smoteWarning = getSmoteWarning();

  function handleSubmit() {
    const params = {
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
    };
    console.log("🔵 [Sidebar] useHpTuning =", params.useHpTuning);
    console.log("🔵 [Sidebar] nIterTuning =", params.nIterTuning);
    console.log("🔵 [AnalysisSidebar] Отправляемые параметры:", params);
    onRun(params);
  }

  if (loading) {
    return (
      <aside className="sidebar">
        <h2>Настройки анализа</h2>
        <p className="muted">⏳ Загрузка данных...</p>
      </aside>
    );
  }
  return (
    <aside className="sidebar">
      <h2> Настройки анализа</h2>
      {/* Индикатор размера данных */}
      {rowCount > 0 && (
        <div
          className={`data-size-indicator ${isLargeDataset ? "large" : ""}`}
          style={{
            padding: "8px",
            borderRadius: "6px",
            marginBottom: "12px",
            background: isLargeDataset
              ? "rgba(255, 100, 100, 0.1)"
              : "var(--bg-secondary)",
            borderLeft: isLargeDataset ? "3px solid #ff6b6b" : "none",
          }}
        >
          <span> Размер данных: </span>
          <strong>{rowCount.toLocaleString()}</strong>
          <span> строк</span>
          {isLargeDataset && (
            <span style={{ color: "#ff6b6b", marginLeft: "8px" }}>
              (большой датасет)
            </span>
          )}
        </div>
      )}
      <label>
        Число кластеров: <b>{nClusters}</b>
        <input
          type="range"
          min="2"
          max="6"
          value={nClusters}
          onChange={(e) => setNClusters(Number(e.target.value))}
        />
      </label>

      <label>
        Порог риска: <b>{riskThreshold.toFixed(2)}</b>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={riskThreshold}
          onChange={(e) => setRiskThreshold(Number(e.target.value))}
        />
      </label>

      <label>
        Порог корреляции: <b>{corrThreshold.toFixed(2)}</b>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={corrThreshold}
          onChange={(e) => setCorrThreshold(Number(e.target.value))}
        />
      </label>

      <label>
        Признаков в финальной модели: <b>{nFeatures}</b>
        <input
          type="range"
          min="3"
          max="15"
          value={nFeatures}
          onChange={(e) => setNFeatures(Number(e.target.value))}
        />
      </label>

      <label>
        Кол-во факторов в SHAP: <b>{shapTopN}</b>
        <input
          type="range"
          min="3"
          max="10"
          value={shapTopN}
          onChange={(e) => setShapTopN(Number(e.target.value))}
        />
      </label>

      <label className="checkbox-label">
        <div style={{ marginBottom: 4, fontSize: 13, fontWeight: 600 }}>
          Приоритетная метрика:
        </div>
        <select
          value={optimizationMetric}
          onChange={(e) => setOptimizationMetric(e.target.value)}
          style={{
            width: "100%",
            padding: 4,
            borderRadius: 4,
            border: "1px solid var(--border)",
            background: "var(--bg)",
            color: "var(--text)",
          }}
        >
          <option value="default">По умолчанию (F1)</option>
          <option value="f1">F1-score</option>
          <option value="roc_auc">ROC-AUC</option>
          <option value="precision">Precision</option>
          <option value="recall">Recall</option>
        </select>
      </label>

      <div style={{ marginTop: 12, marginBottom: 8, fontWeight: 600 }}>
        Модели для обучения:
      </div>
      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={useLR}
          onChange={(e) => setUseLR(e.target.checked)}
        />
        Logistic Regression
      </label>
      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={useRF}
          onChange={(e) => setUseRF(e.target.checked)}
        />
        Random Forest
      </label>

      {/* Предупреждение для RF */}
      {rfWarning && (
        <div
          style={{
            marginLeft: "24px",
            marginTop: "4px",
            marginBottom: "8px",
            padding: "6px 10px",
            borderRadius: "6px",
            fontSize: "12px",
            background:
              rfWarning.level === "error"
                ? "rgba(255, 100, 100, 0.15)"
                : "rgba(255, 193, 7, 0.15)",
            borderLeft:
              rfWarning.level === "error"
                ? "3px solid #ff6b6b"
                : "3px solid #ffc107",
          }}
        >
          {rfWarning.message}
        </div>
      )}
      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={useXGB}
          onChange={(e) => setUseXGB(e.target.checked)}
        />
        XGBoost
      </label>

      <label className="checkbox-label" style={{ marginTop: 8 }}>
        <input
          type="checkbox"
          checked={useSmote}
          onChange={(e) => setUseSmote(e.target.checked)}
        />
        SMOTE (балансировка классов)
      </label>
      {smoteWarning && (
        <div
          style={{
            marginLeft: "24px",
            marginTop: "4px",
            padding: "6px 10px",
            borderRadius: "6px",
            fontSize: "12px",
            background: "rgba(255, 193, 7, 0.15)",
            borderLeft: "3px solid #ffc107",
          }}
        >
          {smoteWarning.message}
        </div>
      )}

      <label className="checkbox-label">
        <input
          type="checkbox"
          checked={useHpTuning}
          onChange={(e) => setUseHpTuning(e.target.checked)}
        />
        Оптимизация гиперпараметров (XGB)
      </label>

      {useHpTuning && (
        <label>
          Итераций тюнинга: <b>{nIterTuning}</b>
          <input
            type="range"
            min="10"
            max="50"
            value={nIterTuning}
            onChange={(e) => setNIterTuning(Number(e.target.value))}
          />
        </label>
      )}

      <button
        className="primary"
        onClick={handleSubmit}
        disabled={!canRun}
        style={{ marginTop: 12 }}
      >
        {busy
          ? "Анализ..."
          : !canRun
            ? "Загрузите данные и выберите цель"
            : "Запустить анализ"}
      </button>

      <p className="hint">
        {!canRun && !busy
          ? "Нужны загруженные данные и подтверждённая целевая переменная (или дождитесь окончания текущего запроса)."
          : "Настройки не влияют на данные, пока не нажата кнопка «Запустить анализ»."}
      </p>
    </aside>
  );
}
