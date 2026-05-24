import { useEffect, useMemo, useState, lazy, useRef } from "react";
import { HashRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import SafeErrorBoundary from "./components/SafeErrorBoundary";
import DatasetHistory from "./components/DatasetHistory";
import {
  healthcheck,
  getExcelPreview,
  processExcel,
  trainAsyncJson,
  getTaskStatus,
  saveExperiment,
  getFullAnalysisStatus,
  runFullAnalysisAsync,
  cancelFullAnalysis,
  runCorrelationAsync,
  getCorrelationStatus,
  processCSV,
} from "./api";
const AnalysisSidebar = lazy(() => import("./pages/AnalysisSidebar"));
const AnalysisResults = lazy(() => import("./pages/AnalysisResults"));
const SheetMapper = lazy(() => import("./pages/SheetMapper"));
const DataEnrichment = lazy(() => import("./pages/DataEnrichment"));
const Imputation = lazy(() => import("./pages/Imputation"));
const DriftCheck = lazy(() => import("./pages/DriftCheck"));
const Crosstab = lazy(() => import("./pages/Crosstab"));
const TimeSeries = lazy(() => import("./pages/TimeSeries"));
const CompositeScore = lazy(() => import("./pages/CompositeScore"));
const Experiments = lazy(() => import("./pages/Experiments"));
const SubsetSelect = lazy(() => import("./pages/SubsetSelect"));
const FeatureCombinations = lazy(() => import("./pages/FeatureCombinations"));
import { handleImputation } from "./api";
import { useSharedData, useDatasetStore } from "./hooks/useDatasetStore";
import "./styles.css";
import { NAV, EXCLUDE_COLS } from "./utils/constants";
import { filterServiceCols } from "./utils/csvHelpers";

function Tabs() {
  const location = useLocation();
  return (
    <div className="tabs">
      {NAV.map((n) => (
        <Link
          key={n.path}
          to={n.path}
          className={location.pathname === n.path ? "active" : ""}
        >
          {n.label}
        </Link>
      ))}
    </div>
  );
}

/** Кнопка переключения темы */
function ThemeToggle() {
  const [dark, setDark] = useState(
    () => !document.documentElement.classList.contains("light"),
  );

  function toggle() {
    document.documentElement.classList.toggle("light");
    setDark(!document.documentElement.classList.contains("light"));
  }

  return (
    <button className="theme-toggle" onClick={toggle} title="Переключить тему">
      {dark ? "☀️" : "🌙"}
    </button>
  );
}

// App.jsx — универсальный PollingTask
function PollingTask({
  taskId,
  title,
  onStatus,
  getStatusFunc = getTaskStatus,
}) {
  const [task, setTask] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!taskId) return;
    let stopped = false;
    const timer = setInterval(async () => {
      try {
        const status = await getStatusFunc(taskId);
        if (!stopped) setTask(status);
        if (onStatus) onStatus(status);
        if (status.status === "SUCCESS" || status.status === "FAILURE")
          clearInterval(timer);
      } catch (e) {
        if (!stopped) setError(String(e.message || e));
        clearInterval(timer);
      }
    }, 2000);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [taskId, getStatusFunc, onStatus]);

  if (!taskId) return null;
  return (
    <div className="card">
      <h3>{title}</h3>
      <p>
        <b>Task ID:</b> {taskId}
      </p>
      {error ? <p className="error">{error}</p> : null}
      {task ? (
        <>
          <p>
            <b>Статус:</b> {task.status}
          </p>
          <p>
            <b>Этап:</b> {task.stage || "-"} | <b>Прогресс:</b>{" "}
            {task.progress ?? 0}%
          </p>
          {task.result ? (
            <pre>{JSON.stringify(task.result, null, 2)}</pre>
          ) : null}
          {task.error ? <pre className="error">{task.error}</pre> : null}
        </>
      ) : (
        <p>Ожидание обновлений...</p>
      )}
    </div>
  );
}

function MainPage() {
  const [Plot, setPlot] = useState(null);

  const shared = useSharedData();
  const [restored, setRestored] = useState(false);

  const [file, setFile] = useState(null);
  const [csvData, setCsvData] = useState(null);
  const csvDataRef = useRef(null);
  const [refreshFlag, setRefreshFlag] = useState(0);
  const [csvPreview, setCsvPreview] = useState({
    headers: [],
    rows: [],
    rowCount: 0,
  });
  const [isDragging, setIsDragging] = useState(false);
  const [historyRefreshTrigger, setHistoryRefreshTrigger] = useState(0);
  const [analysisStage, setAnalysisStage] = useState("");
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [corrResult, setCorrResult] = useState(null);
  const [trainTaskId, setTrainTaskId] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);
  // Состояние для сохранения эксперимента
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [saveDesc, setSaveDesc] = useState("");
  const [saving, setSaving] = useState(false);
  const [sheetTypeInfo, setSheetTypeInfo] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [analysisTaskId, setAnalysisTaskId] = useState(null);

  const [analysisStatus, setAnalysisStatus] = useState(null);
  const [analysisLoading] = useState(false);

  const [corrTaskId, setCorrTaskId] = useState(null);
  const [corrLoading, setCorrLoading] = useState(false);
  // Эффект для polling
  useEffect(() => {
    console.log("🔵 POLLING EFFECT: analysisTaskId =", analysisTaskId);

    if (!analysisTaskId) return;

    let isActive = true;
    let intervalId = null;

    const poll = async () => {
      if (!isActive) return;

      try {
        const data = await getFullAnalysisStatus(analysisTaskId);

        if (!isActive) return;

        setAnalysisStatus(data.status);

        if (data.status === "SUCCESS") {
          await saveAnalysisResult(analysisTaskId, data.result);
          sessionStorage.setItem("has_unviewed_result", "true");
          sessionStorage.setItem("unviewed_result_id", analysisTaskId);
          console.log("=== ПОЛНЫЙ ОТВЕТ ОТ БЭКЕНДА ===", data.result);
          console.log("=== target_col в ответе ===", data.result?.target_col);
          console.log("🔵 ВРЕМЯ ПОЛУЧЕНИЯ:", new Date().toLocaleTimeString());
          console.log(
            "🔵 РАЗМЕР data.result:",
            JSON.stringify(data.result).length / 1024 / 1024,
            "MB",
          );
          setAnalysisResult(data.result);
          if (data.result?.data_with_clusters) {
            shared.updateData(data.result.data_with_clusters);
          }
          setAnalysisTaskId(null);
          setBusy(false);
          if (intervalId) clearInterval(intervalId);
          sessionStorage.removeItem("active_analysis_task_id");
          sessionStorage.removeItem("active_analysis_params");
          sessionStorage.removeItem("active_analysis_target");
        } else if (data.status === "FAILURE") {
          setError(data.error || "Analysis failed");
          setAnalysisTaskId(null);
          setBusy(false);
          if (intervalId) clearInterval(intervalId);
          sessionStorage.removeItem("active_analysis_task_id");
          sessionStorage.removeItem("active_analysis_params");
          sessionStorage.removeItem("active_analysis_target");
        } else if (data.status === "PROGRESS") {
          setAnalysisStage(data.stage || "Выполняется...");
          setAnalysisProgress(data.progress || 0);
          setAnalysisStatus("PROGRESS");
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    };

    // Первый запрос сразу
    poll();

    // Затем каждые 3 секунды
    intervalId = setInterval(poll, 3000);

    return () => {
      isActive = false;
      if (intervalId) clearInterval(intervalId);
    };
  }, [analysisTaskId]);

  const clearIndexedDB = useDatasetStore((state) => state.clearIndexedDB);
  const saveAnalysisResult = useDatasetStore(
    (state) => state.saveAnalysisResult,
  );
  const loadAnalysisResult = useDatasetStore(
    (state) => state.loadAnalysisResult,
  );
  const deleteAnalysisResult = useDatasetStore(
    (state) => state.deleteAnalysisResult,
  );
  const handleClearCache = async () => {
    if (
      window.confirm(
        "Очистить кеш всех загруженных данных? Это действие необратимо.",
      )
    ) {
      await clearIndexedDB();
      setCsvData(null);
      setCsvPreview({ headers: [], rows: [], rowCount: 0 });
      csvDataRef.current = null;
      alert("Кеш очищен");
    }
  };

  async function onRunAnalysisAsync(params) {
    console.log("🔵 [App] Получены параметры в onRunAnalysisAsync:", params);
    const fullData = csvDataRef.current;
    if (!fullData) return;
    if (!targetColumn) {
      setError("Пожалуйста, выберите целевую переменную на главной вкладке");
      return;
    }
    sessionStorage.removeItem("active_analysis_task_id");
    sessionStorage.removeItem("active_analysis_params");
    sessionStorage.removeItem("active_analysis_target");

    setBusy(true);
    setAnalysisResult(null);
    setAnalysisTaskId(null);
    setError("");

    try {
      const requestParams = {
        ...params,
        target_col: targetColumn,
      };
      console.log("🔵 [App] Отправляем в API:", requestParams);
      console.log("🔵 targetColumn =", targetColumn);
      console.log("🔵 targetSelected =", targetSelected);

      const res = await runFullAnalysisAsync(fullData, requestParams);
      const taskId = res.task_id;
      setAnalysisTaskId(taskId);

      sessionStorage.setItem("active_analysis_task_id", taskId);
      sessionStorage.setItem("active_analysis_target", targetColumn);
      sessionStorage.setItem(
        "active_analysis_params",
        JSON.stringify({
          target_col: targetColumn,
          ...params,
        }),
      );
    } catch (e) {
      console.error("Failed to start analysis:", e);
      setError(e.message);
      setBusy(false);
    }
  }
  // Отображение статуса
  const renderAnalysisStatus = () => {
    if (!analysisTaskId) return null;

    const handleCancel = async () => {
      if (window.confirm("Отменить выполнение анализа?")) {
        try {
          await cancelFullAnalysis(analysisTaskId);
          setAnalysisTaskId(null);
          setBusy(false);
          setError("Анализ отменён пользователем");
          sessionStorage.removeItem("active_analysis_task_id");
          sessionStorage.removeItem("active_analysis_params");
          sessionStorage.removeItem("active_analysis_target");
        } catch (err) {
          console.error("Cancel failed:", err);
          setError("Не удалось отменить задачу");
        }
      }
    };

    return (
      <div className="card" style={{ marginTop: 12 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <h3 style={{ margin: 0 }}>Статус анализа</h3>
          <button
            onClick={handleCancel}
            style={{
              padding: "4px 12px",
              background: "#e74c3c",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            Отменить
          </button>
        </div>

        {/* Прогресс-бар */}
        <div className="progress-container" style={{ marginBottom: 12 }}>
          <div
            className="progress-bar-fill"
            style={{
              width: `${analysisProgress}%`,
              backgroundColor:
                analysisStatus === "FAILURE" ? "#e74c3c" : "var(--primary)",
              height: "8px",
              borderRadius: "4px",
              transition: "width 0.3s ease",
            }}
          />
        </div>

        {/* Статус и этап */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            flexWrap: "wrap",
          }}
        >
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: "bold" }}>
              {analysisStage || "Выполняется..."}
            </div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
              Прогресс: {analysisProgress}%
            </div>
          </div>
          {analysisProgress > 0 &&
            analysisProgress < 100 &&
            analysisStatus !== "FAILURE" && (
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                Пожалуйста, подождите...
              </div>
            )}
        </div>

        {/* Ошибка если есть */}
        {analysisStatus === "FAILURE" && (
          <div style={{ marginTop: 8, color: "#e74c3c", fontSize: 13 }}>
            {analysisStage || "Ошибка выполнения"}
          </div>
        )}
      </div>
    );
  };

  useEffect(() => {
    if (!restored && shared.datasetId) {
      setRestored(true);
    }
  }, [shared, restored]);

  useEffect(() => {
    import("react-plotly.js").then((module) => setPlot(() => module.default));
  }, []);

  // Восстановление из shared dataset при монтировании (переключение вкладок / F5)
  useEffect(() => {
    const loadPreview = async () => {
      const previewRows = await shared.getPreview(5);
      if (previewRows.length) {
        const headers = Object.keys(previewRows[0]);
        setCsvPreview({
          headers,
          rows: previewRows.map((r) => headers.map((h) => String(r[h] ?? ""))),
          rowCount: shared.metadata.rowCount || previewRows.length,
        });
      }
    };

    if (shared.hasShared) {
      loadPreview();
    }
  }, [shared, shared.hasShared, shared.metadata.rowCount]);

  // Excel Mapping state
  const [sheetPreview, setSheetPreview] = useState(null); // Данные для SheetMapper
  const [mappingConfig, setMappingConfig] = useState(null); // Конфиг для обработки
  const [excelSheets, setExcelSheets] = useState([]); // Имена листов в файле
  const [selectedSheet, setSelectedSheet] = useState(""); // Выбранный лист
  const [rawExcelData, setRawExcelData] = useState(null); // Сырые данные после processExcel (до обогащения)

  // Выбор целевой переменной
  const [targetColumn, setTargetColumn] = useState("");
  const [targetSelected, setTargetSelected] = useState(false);

  // Доступные колонки для выбора target
  const targetCandidates = useMemo(() => {
    const data = csvDataRef.current;
    if (!data?.length) return [];
    const allKeys = Object.keys(data[0]);
    console.log("[targetCandidates] allKeys:", allKeys);
    console.log("[targetCandidates] EXCLUDE_COLS:", [...EXCLUDE_COLS]);
    const filtered = allKeys.filter((k) => !EXCLUDE_COLS.has(k.toLowerCase()));
    console.log("[targetCandidates] filtered:", filtered);
    return filtered;
  }, [EXCLUDE_COLS, refreshFlag]);

  useEffect(() => {
    console.log("=== ОТЛАДКА ПОСЛЕ ОБНОВЛЕНИЯ ===");
    console.log("csvDataRef.current?.length:", csvDataRef.current?.length);
    console.log("targetCandidates:", targetCandidates);
    console.log("targetSelected:", targetSelected);
    console.log("sheetPreview:", sheetPreview);
    console.log("rawExcelData:", rawExcelData);
    console.log(
      "Условие показа:",
      csvData?.length > 0 &&
        !targetSelected &&
        targetCandidates?.length > 0 &&
        !sheetPreview &&
        !rawExcelData,
    );
  }, [csvData, targetSelected, targetCandidates, sheetPreview, rawExcelData]);

  useEffect(() => {
    if (analysisTaskId) {
      sessionStorage.setItem("active_analysis_task_id", analysisTaskId);
      // Сохраняем параметры для возможного восстановления
      sessionStorage.setItem("active_analysis_target", targetColumn);
    }
  }, [analysisTaskId, targetColumn]);

  // Восстанавливаем задачу при монтировании компонента
  useEffect(() => {
    const restoreState = async () => {
      console.log("🔵 RESTORE: анализ состояния");
      console.log("  - analysisResult:", analysisResult);
      console.log("  - analysisTaskId:", analysisTaskId);
      // 1. Проверяем завершённый, но непросмотренный результат
      const hasUnviewed = sessionStorage.getItem("has_unviewed_result");
      const unviewedId = sessionStorage.getItem("unviewed_result_id");
      console.log("  - has_unviewed_result:", hasUnviewed);
      console.log("  - unviewed_result_id:", unviewedId);

      if (
        hasUnviewed === "true" &&
        !analysisResult &&
        !analysisTaskId &&
        unviewedId
      ) {
        console.log(
          "🔵 RESTORE: пытаюсь загрузить результат из IndexedDB, id=",
          unviewedId,
        );
        const result = await loadAnalysisResult(unviewedId);
        console.log(
          "🔵 RESTORE: загруженный результат:",
          result ? "есть" : "null",
        );
        if (result) {
          console.log("🔵 RESTORE: результат успешно загружен");
          setAnalysisResult(result);
          sessionStorage.removeItem("has_unviewed_result");
          sessionStorage.removeItem("unviewed_result_id");
          await deleteAnalysisResult(unviewedId);
          console.log("🔵 RESTORE: флаги очищены");
          return;
        } else {
          console.log("🔵 RESTORE: результат не найден, очищаю флаги");
          sessionStorage.removeItem("has_unviewed_result");
          sessionStorage.removeItem("unviewed_result_id");
        }
      }

      const savedTaskId = sessionStorage.getItem("active_analysis_task_id");
      const savedTarget = sessionStorage.getItem("active_analysis_target");
      console.log("🔵 RESTORE EFFECT: savedTaskId =", savedTaskId);
      if (savedTaskId && !analysisTaskId && !analysisResult) {
        console.log("Восстановление задачи:", savedTaskId);
        setAnalysisTaskId(savedTaskId);
        setBusy(true);
        if (savedTarget) {
          setTargetColumn(savedTarget);
          setTargetSelected(true);
        }
      }
    };
    restoreState();
  }, [analysisResult, analysisTaskId]);

  // Эффект для опроса статуса корреляции
  useEffect(() => {
    if (!corrTaskId) return;

    let isActive = true;
    let intervalId = null;

    const poll = async () => {
      if (!isActive) return;
      try {
        const data = await getCorrelationStatus(corrTaskId);
        console.log("🔵 Correlation status:", data);

        if (data.status === "SUCCESS") {
          console.log("🔵 Correlation result:", data.result);
          setCorrResult(data.result);
          setCorrTaskId(null);
          setCorrLoading(false);
          if (intervalId) clearInterval(intervalId);
        } else if (data.status === "FAILURE") {
          setError(data.error || "Correlation failed");
          setCorrTaskId(null);
          setCorrLoading(false);
          if (intervalId) clearInterval(intervalId);
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    };

    poll();
    intervalId = setInterval(poll, 2000);

    return () => {
      isActive = false;
      if (intervalId) clearInterval(intervalId);
    };
  }, [corrTaskId]);

  // Статистика по выбранной колонке
  const targetStats = useMemo(() => {
    if (!targetColumn) return null;
    const data = csvDataRef.current;
    if (!data?.length) return null;

    const sampleSize = Math.min(data.length, 500);
    const vals = [];
    for (let i = 0; i < sampleSize; i++) {
      const val = data[i][targetColumn];
      if (val !== "" && val != null) vals.push(val);
    }

    // Определяем тип на основе сэмпла
    const isNumeric = vals.every((v) => typeof v === "number");
    const unique = new Set(vals).size;
    return {
      type: isNumeric ? "числовой" : "категориальный",
      unique,
      min: isNumeric ? Math.min(...vals).toFixed(2) : null,
      max: isNumeric ? Math.max(...vals).toFixed(2) : null,
      needsBinarize: isNumeric && unique > 2,
    };
  }, [targetColumn]);

  /** Единое условие «можно запускать полный анализ» — сайдбар и будущие проверки опираются на это. */
  const canRun = useMemo(
    () => Boolean(csvData?.length) && targetSelected && !busy,
    [csvData, targetSelected, busy],
  );

  async function onCorrelation() {
    if (!file) return;

    setCorrLoading(true);
    setCorrResult(null);
    setError("");

    try {
      const res = await runCorrelationAsync(csvDataRef.current, targetColumn);
      setCorrTaskId(res.task_id);
    } catch (e) {
      setError(String(e.message || e));
      setCorrLoading(false);
    }
  }
  async function onTrainAsync() {
    if (!csvData || !csvData.length) return;
    if (!targetColumn) {
      setError("Пожалуйста, выберите целевую переменную на главной вкладке");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const res = await trainAsyncJson(csvData, targetColumn);
      setTrainTaskId(res.task_id);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function handleSaveExperiment() {
    if (!saveName.trim() || !analysisResult) return;
    setSaving(true);
    try {
      const fullMetrics = {
        test: analysisResult.test_metrics || {},
        cv_results: analysisResult.cv_results || {},
        ...analysisResult.metrics,
      };

      const fullConfig = {
        ...(analysisResult.config || {}),
        target_col: analysisResult.target_col,
        model_name:
          analysisResult.config?.model_name || analysisResult.model_name,
        test_metrics: analysisResult.test_metrics,
        n_samples: analysisResult.config?.n_samples || analysisResult.n_samples,
        n_features: analysisResult.selected_features?.length || 0,
        n_clusters:
          analysisResult.config?.n_clusters || analysisResult.n_clusters,
        use_smote: analysisResult.config?.use_smote,
        corr_threshold: analysisResult.config?.corr_threshold,
        optimization_metric: analysisResult.config?.optimization_metric,
        shap_top_n: analysisResult.config?.shap_top_n,
        risk_threshold: analysisResult.config?.risk_threshold,
        use_lr: analysisResult.config?.use_lr,
        use_rf: analysisResult.config?.use_rf,
        use_xgb: analysisResult.config?.use_xgb,
        use_hp_tuning: analysisResult.config?.use_hp_tuning || false,
        n_iter_tuning: analysisResult.config?.n_iter_tuning || 20,
        timestamp: new Date().toISOString(),
      };
      console.log(" Сохраняемый конфиг:", fullConfig);
      await saveExperiment(
        saveName,
        fullMetrics,
        analysisResult.selected_features || [],
        saveDesc,
        fullConfig,
      );
      console.log(" analysisResult.config:", analysisResult.config);
      console.log("  analysisResult full:", analysisResult);
      console.log("  analysisResult.model_name:", analysisResult.model_name);
      console.log(
        "  analysisResult.config.model_name:",
        analysisResult.config?.model_name,
      );

      setSaveModalOpen(false);
      setSaveName("");
      setSaveDesc("");
      alert("  Эксперимент сохранён!");
    } catch (e) {
      setError("Ошибка сохранения: " + e.message);
    } finally {
      setSaving(false);
    }
  }

  /** Вспомогательная функция для обновления превью и данных */
  function setDataAndPreview(data) {
    if (!data || !data.length) {
      console.log("  setDataAndPreview: нет данных, выход");
      return;
    }
    console.log("  setDataAndPreview: начало, data.length =", data.length);

    let filtered = filterServiceCols(data, EXCLUDE_COLS);

    console.log(
      "  setDataAndPreview: после filterServiceCols, filtered.length =",
      filtered.length,
    );
    console.log(
      "  setDataAndPreview: колонки filtered:",
      Object.keys(filtered[0] || {}),
    );

    csvDataRef.current = filtered;
    setCsvData(filtered.slice(0, 100));
    console.log(
      "  setDataAndPreview: csvData установлен, длина =",
      filtered.slice(0, 100).length,
    );
    shared.updateData(filtered);
    setRefreshFlag((prev) => prev + 1);

    // Обновляем превью
    if (filtered.length > 0) {
      const headers = Object.keys(filtered[0]);
      const previewRows = filtered
        .slice(0, 5)
        .map((r) => headers.map((h) => String(r[h] ?? "")));
      setCsvPreview({ headers, rows: previewRows, rowCount: filtered.length });
      console.log(
        "  setDataAndPreview: превью установлено, rowCount:",
        filtered.length,
      );
      console.log(
        "  setDataAndPreview: превью установлено, headers =",
        headers,
      );
    } else {
      setCsvPreview({ headers: [], rows: [], rowCount: 0, riskPct: null });
    }
    console.log("[setDataAndPreview] filtered length:", filtered.length);
    console.log(
      "[setDataAndPreview] filtered columns:",
      Object.keys(filtered[0] || {}),
    );
    console.log(
      "[setDataAndPreview] csvDataRef.current length:",
      csvDataRef.current?.length,
    );
    setHistoryRefreshTrigger((prev) => prev + 1);
    console.log("  setDataAndPreview: csvData установлен?", !!csvData);
    console.log(
      "  setDataAndPreview: rawExcelData должен быть null, но сейчас:",
      rawExcelData,
    );
  }

  /** Обработчик подтверждения маппинга из SheetMapper */
  async function onMappingConfirm(config) {
    if (!file) return;
    setBusy(true);
    setMappingConfig(config);

    try {
      const group = sheetPreview?.detected_group || "numeric";
      const res = await processExcel(
        file,
        sheetPreview?.sheet_name || "0",
        group,
        config,
      );
      // Сохраняем сырые данные и показываем DataEnrichment
      setRawExcelData(res.data);
    } catch (e) {
      setError("Ошибка обработки Excel: " + e.message);
    } finally {
      setBusy(false);
    }
  }

  /** Обработчик подтверждения обогащения */
  async function onEnrichmentConfirm({ strategy, threshold }) {
    if (!rawExcelData) return;
    setBusy(true);
    setError("");
    try {
      const res = await handleImputation(rawExcelData, strategy, threshold);
      console.log(
        "  handleEnrichmentConfirm: данные получены",
        res.data?.length,
      );
      console.log("  КОЛОНКИ В ОТВЕТЕ:", Object.keys(res.data[0] || {}));
      console.log(
        "  res.data тип:",
        Array.isArray(res.data) ? "массив" : typeof res.data,
      );
      console.log("  res.data первые 2 элемента:", res.data?.slice(0, 2));
      setDataAndPreview(res.data);
      console.log("  Сброс состояний...");
      setRawExcelData(null);
      setSheetPreview(null);
      setSheetTypeInfo(null);
      setTargetSelected(false);
      console.log(
        "  После сброса: rawExcelData=null, sheetPreview=null, sheetTypeInfo=null",
      );
    } catch (e) {
      setError("Ошибка обогащения: " + e.message);
    } finally {
      setBusy(false);
    }
  }

  /** Обработчик пропуска обогащения */
  function onEnrichmentSkip() {
    if (!rawExcelData) return;
    console.log(
      "  handleEnrichmentSkip: пропускаем обогащение",
      rawExcelData.length,
    );
    console.log(
      "  res.data тип:",
      Array.isArray(rawExcelData.data) ? "массив" : typeof rawExcelData.data,
    );
    console.log(
      "  res.data первые 2 элемента:",
      rawExcelData.data?.slice(0, 2),
    );
    setDataAndPreview(rawExcelData);

    console.log("  Сброс состояний...");
    setRawExcelData(null);
    setSheetPreview(null);
    setSheetTypeInfo(null);
    setTargetSelected(false);

    console.log(
      "  После сброса: rawExcelData=null, sheetPreview=null, sheetTypeInfo=null",
    );
  }

  /** Загружает превью выбранного листа Excel */
  async function loadSheetPreview(fileObj, sheetName) {
    if (!fileObj || !sheetName) return;

    setBusy(true);
    setError("");
    setRawExcelData(null);
    setMappingConfig(null);
    try {
      console.log("=== loadSheetPreview ===");
      console.log("Выбранный лист:", sheetName);

      const preview = await getExcelPreview(fileObj, sheetName);
      console.log("Превью от сервера:", preview);
      console.log(
        "Все колонки:",
        preview.columns?.map((c) => ({
          name: c.name,
          dtype: c.dtype,
          isString: c.dtype === "object" || c.dtype === "string",
        })),
      );
      setSheetPreview(preview);
      setSheetTypeInfo({
        group_label: preview.group_label,
        detected_group: preview.detected_group,
      });
      const isNumericOnly = preview.detected_group === "numeric";
      if (isNumericOnly) {
        // Для чисто числовых листов сразу обрабатываем без маппера
        console.log("Числовой лист — пропускаем маппинг и сразу обрабатываем");
        const group = preview.detected_group || "numeric";
        const res = await processExcel(fileObj, sheetName, group, null);
        setRawExcelData(res.data);
        setSheetPreview(null); // Не показываем маппер
      }
      setBusy(false);
    } catch (e) {
      setError("Ошибка превью: " + e.message);
      setBusy(false);
      setSheetPreview(null);
    }
  }

  /** Выбор листа Excel */
  function onSheetSelect(name) {
    setSelectedSheet(name);
    setSheetPreview(null);
    setMappingConfig(null);
    setRawExcelData(null);
    setCsvData(null);
    setCsvPreview({ headers: [], rows: [], rowCount: 0, riskPct: null });
    loadSheetPreview(file, name);
  }
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (
      file &&
      (file.name.endsWith(".csv") ||
        file.name.endsWith(".xlsx") ||
        file.name.endsWith(".xls"))
    ) {
      // Создаём синтетическое событие для onFileChange
      const syntheticEvent = { target: { files: [file] } };
      onFileChange(syntheticEvent);
    } else {
      setError("Пожалуйста, загрузите файл CSV или Excel");
    }
  };
  /** Обрабатывает загрузку CSV или Excel */
  async function onFileChange(e) {
    const next = e.target.files?.[0] || null;
    setFile(next);
    setCorrResult(null);
    setAnalysisResult(null);
    setCsvData(null);
    if (csvDataRef) {
      csvDataRef.current = null;
    }
    setSheetPreview(null);
    setMappingConfig(null);
    setRawExcelData(null);
    setExcelSheets([]);
    setSelectedSheet("");
    setTargetColumn("");
    setTargetSelected(false);

    if (!next) return;

    const isExcel = next.name.endsWith(".xlsx") || next.name.endsWith(".xls");

    try {
      let allRows = [];

      if (isExcel) {
        // ========== EXCEL: читаем листы ==========
        const XLSX = await import("xlsx");
        const buf = await next.arrayBuffer();
        const wb = XLSX.read(buf, { type: "array" });

        const sheetNames = wb.SheetNames;
        if (sheetNames.length > 1) {
          setExcelSheets(sheetNames);
          setCsvPreview({ headers: [], rows: [], rowCount: 0 });
          return;
        }

        // Один лист — читаем сразу
        const sheet = wb.Sheets[sheetNames[0]];
        allRows = XLSX.utils.sheet_to_json(sheet, { defval: null });
        setRawExcelData(allRows);
        setCsvPreview({
          headers: Object.keys(allRows[0] || {}),
          rows: allRows
            .slice(0, 5)
            .map((r) => Object.values(r).map((v) => String(v))),
          rowCount: allRows.length,
          riskPct: null,
        });
        setSheetTypeInfo({
          group_label: "  Данные",
          detected_group: "numeric",
        });
      } else {
        // ========== CSV: читаем напрямую ==========
        const res = await processCSV(next);
        setRawExcelData(res.data);
        setCsvPreview({
          headers: Object.keys(res.data[0] || {}),
          rows: res.data
            .slice(0, 5)
            .map((r) => Object.values(r).map((v) => String(v))),
          rowCount: res.rows,
          riskPct: null,
        });
        setSheetTypeInfo({
          group_label: "Числовые данные (csv)",
          detected_group: "numeric",
        });
      }
    } catch (err) {
      console.error(err);
      setError("Ошибка чтения файла: " + err.message);
      setBusy(false);
    }
  }

  function renderCorrelationTable() {
    const matrix = corrResult?.correlation_matrix;

    if (!matrix) {
      // Если есть correlations с целевой переменной
      if (corrResult?.correlations) {
        return (
          <div className="table-wrap">
            <table className="matrix">
              <thead>
                <tr>
                  <th>Признак</th>
                  <th>Корреляция с {corrResult.target_col}</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(corrResult.correlations).map(
                  ([feature, corr]) => (
                    <tr key={feature}>
                      <td>
                        <b>{feature}</b>
                      </td>
                      <td
                        className={
                          corr > 0
                            ? "heat-high"
                            : corr < 0
                              ? "heat-neg-mid"
                              : ""
                        }
                      >
                        {corr.toFixed(4)}
                      </td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        );
      }
      return <p className="muted">Нет данных для отображения</p>;
    }
    const cols = Object.keys(matrix);
    const getCellClass = (v) => {
      if (v > 0.7) return "heat-high";
      if (v > 0.3) return "heat-mid";
      if (v < -0.7) return "heat-neg-high";
      if (v < -0.3) return "heat-neg-mid";
      return "heat-low";
    };
    return (
      <div className="table-wrap">
        <table className="matrix">
          <thead>
            <tr>
              <th>feature</th>
              {cols.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cols.map((r) => (
              <tr key={r}>
                <td>
                  <b>{r}</b>
                </td>
                {cols.map((c) => {
                  const v = Number(matrix[r]?.[c] ?? 0);
                  return (
                    <td key={`${r}_${c}`} className={getCellClass(v)}>
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div className="analysis-layout">
      <AnalysisSidebar onRun={onRunAnalysisAsync} busy={busy} canRun={canRun} />

      <div className="analysis-main">
        <div className="card">
          <h2>1) Загрузка данных</h2>
          <div
            className="drag-drop-area"
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            style={{
              border: `2px dashed ${isDragging ? "var(--primary)" : "var(--border)"}`,
              borderRadius: 8,
              padding: "20px",
              textAlign: "center",
              cursor: "pointer",
              backgroundColor: isDragging
                ? "var(--bg-secondary)"
                : "transparent",
              transition: "all 0.2s ease",
            }}
            onClick={() => document.getElementById("fileInput").click()}
          >
            <input
              id="fileInput"
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={onFileChange}
              style={{ display: "none" }}
            />
            <div> Нажмите или перетащите файл сюда</div>
            <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
              Поддерживаются .csv, .xlsx, .xls
            </div>
          </div>
          <p>Файл: {file?.name || "не выбран"}</p>
          {mappingConfig && (
            <p className="muted" style={{ marginTop: 4 }}>
              Маппинг Excel: лист «{mappingConfig.sheet_name ?? "—"}», колонок с
              правилами:{" "}
              <b>{Object.keys(mappingConfig.columns || {}).length}</b>
              {mappingConfig.detected_group ? (
                <>
                  {" "}
                  · тип листа: <b>{mappingConfig.detected_group}</b>
                </>
              ) : null}
            </p>
          )}

          {/* --- ВЫБОР ЛИСТА EXCEL --- */}
          {excelSheets.length > 1 && (
            <div
              className="sheet-selector"
              style={{
                marginTop: 12,
                padding: 12,
                background: "var(--bg-secondary)",
                borderRadius: 8,
              }}
            >
              <label>
                <b>Файл содержит {excelSheets.length} листов. Выберите:</b>
                <select
                  value={selectedSheet}
                  onChange={(e) => onSheetSelect(e.target.value)}
                  style={{
                    width: "100%",
                    marginTop: 6,
                    padding: 8,
                    borderRadius: 6,
                    border: "1px solid var(--border)",
                    background: "var(--bg)",
                    color: "var(--text)",
                  }}
                >
                  <option value="">— Выберите лист —</option>
                  {excelSheets.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}
          {/* ------------------------------- */}

          {busy && <p> Загрузка и анализ файла...</p>}

          {/* --- ИНТЕГРАЦИЯ SHEET MAPPER --- */}
          {sheetPreview && !rawExcelData && (
            <SheetMapper
              preview={sheetPreview}
              onConfirm={onMappingConfirm}
              onSkip={() => {
                onMappingConfirm(null);
              }}
            />
          )}
          {/* ------------------------------- */}

          {/* --- ОБОГАЩЕНИЕ ДАННЫХ --- */}
          {rawExcelData && !csvData && (
            <DataEnrichment
              groupLabel={sheetTypeInfo?.group_label}
              detectedGroup={sheetTypeInfo?.detected_group}
              onConfirm={onEnrichmentConfirm}
              onSkip={onEnrichmentSkip}
              isLoading={busy}
            />
          )}
          {/* --------------------------- */}

          {csvPreview.rowCount > 0 && (
            <>
              <p>
                Строк: <b>{csvPreview.rowCount}</b>
                {csvPreview.riskPct != null && (
                  <>
                    {" "}
                    | Доля risk_flag=1: <b>{csvPreview.riskPct.toFixed(1)}%</b>
                  </>
                )}
              </p>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      {csvPreview.headers.map((h) => (
                        <th key={h}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {csvPreview.rows.map((r, i) => (
                      <tr key={i}>
                        {r.map((c, j) => (
                          <td key={`${i}_${j}`}>{c}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
        {/*=== выбор датасета ===*/}
        <DatasetHistory
          onLoad={(data, datasetId) => {
            csvDataRef.current = data;
            setCsvData(data.slice(0, 100));
            setCsvPreview({
              headers: Object.keys(data[0] || {}),
              rows: data
                .slice(0, 5)
                .map((r) => Object.values(r).map((v) => String(v))),
              rowCount: data.length,
              riskPct: null,
            });
            setRefreshFlag((prev) => prev + 1);
            useDatasetStore.getState().setCurrentDatasetId(datasetId);
            setTargetSelected(false);
            setTargetColumn("");
            setSheetPreview(null);
            setRawExcelData(null);
            setExcelSheets([]);
            setSelectedSheet("");
            setAnalysisResult(null);
            setCorrResult(null);
            setHistoryRefreshTrigger((prev) => prev + 1);
          }}
          refreshTrigger={historyRefreshTrigger}
        />
        {/* === ВЫБОР ЦЕЛЕВОЙ ПЕРЕМЕННОЙ === */}
        {csvData &&
          csvData.length > 0 &&
          !targetSelected &&
          targetCandidates.length > 0 &&
          !sheetPreview &&
          !rawExcelData && (
            <div className="card">
              <h2>Выбор целевой переменной</h2>
              <p>
                Всего записей: <b>{csvData.length}</b> | Колонки:{" "}
                <b>{targetCandidates.length}</b>
              </p>

              <label>
                <b>Колонка для предсказания:</b>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  style={{
                    width: "100%",
                    padding: 8,
                    marginTop: 4,
                    borderRadius: 6,
                    border: "1px solid var(--border)",
                    background: "var(--bg)",
                    color: "var(--text)",
                  }}
                >
                  <option value="">— Выберите колонку —</option>
                  {targetCandidates.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>

              {targetStats && (
                <div className="muted" style={{ fontSize: 13, marginTop: 8 }}>
                  <p style={{ margin: "4px 0" }}>
                    Тип: <b>{targetStats.type}</b> | Уникальных:{" "}
                    <b>{targetStats.unique}</b>
                    {targetStats.min != null && (
                      <>
                        {" "}
                        | Диапазон: {targetStats.min} — {targetStats.max}
                      </>
                    )}
                  </p>
                  {targetStats.needsBinarize && (
                    <p className="muted" style={{ color: "var(--primary)" }}>
                      Колонка содержит более 2 значений — будет преобразована в
                      бинарную (порог = медиана)
                    </p>
                  )}
                </div>
              )}

              <div className="row" style={{ marginTop: 12 }}>
                <button
                  className="primary"
                  disabled={!targetColumn}
                  onClick={() => setTargetSelected(true)}
                >
                  Подтвердить выбор цели
                </button>
                <button
                  onClick={() => {
                    setCsvData(null);
                    setFile(null);
                    setTargetColumn("");
                    setTargetSelected(false);
                    setSheetPreview(null);
                    setCsvPreview({
                      headers: [],
                      rows: [],
                      rowCount: 0,
                      riskPct: null,
                    });
                    csvDataRef.current = null;
                    setRawExcelData(null);
                    setExcelSheets([]);
                    setSelectedSheet("");
                    setError("");
                    setCorrResult(null);
                    setAnalysisResult(null);
                    shared.clearData();
                  }}
                >
                  Сбросить данные
                </button>
              </div>
            </div>
          )}
        {error && <p className="error">{error}</p>}
        {renderAnalysisStatus()}
        {/* Результаты ПОСЛЕ нажатия кнопки */}
        {analysisResult && (
          <>
            <AnalysisResults result={analysisResult} />
            <div className="card" style={{ marginTop: 16 }}>
              <button
                type="button"
                className="primary"
                onClick={() => setSaveModalOpen(true)}
              >
                Сохранить как эксперимент…
              </button>
              <button
                onClick={handleClearCache}
                style={{ background: "#e74c3c", color: "white" }}
              >
                Очистить кеш данных
              </button>
            </div>
          </>
        )}

        {saveModalOpen && (
          <div
            className="modal-backdrop"
            role="presentation"
            onClick={() => {
              if (!saving) setSaveModalOpen(false);
            }}
          >
            <div
              className="modal-panel"
              role="dialog"
              aria-modal="true"
              aria-labelledby="save-experiment-title"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 id="save-experiment-title"> Сохранить анализ</h3>
              <p className="muted" style={{ marginTop: 0 }}>
                Имя и описание попадут в каталог экспериментов (вкладка
                «Эксперименты»).
              </p>
              <label style={{ display: "block", marginTop: 12 }}>
                Название:
                <input
                  type="text"
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                  placeholder="Например: Вильямс (SMOTE выкл)"
                  style={{
                    width: "100%",
                    marginTop: 4,
                    padding: 8,
                    borderRadius: 6,
                    border: "1px solid var(--border)",
                    background: "var(--bg)",
                    color: "var(--text)",
                  }}
                />
              </label>
              <label style={{ display: "block", marginTop: 12 }}>
                Описание:
                <input
                  type="text"
                  value={saveDesc}
                  onChange={(e) => setSaveDesc(e.target.value)}
                  placeholder="Комментарий…"
                  style={{
                    width: "100%",
                    marginTop: 4,
                    padding: 8,
                    borderRadius: 6,
                    border: "1px solid var(--border)",
                    background: "var(--bg)",
                    color: "var(--text)",
                  }}
                />
              </label>
              <div
                className="row"
                style={{ marginTop: 16, justifyContent: "flex-end", gap: 8 }}
              >
                <button
                  type="button"
                  onClick={() => setSaveModalOpen(false)}
                  disabled={saving}
                >
                  Отмена
                </button>
                <button
                  type="button"
                  className="primary"
                  onClick={handleSaveExperiment}
                  disabled={!saveName.trim() || saving}
                >
                  {saving ? "…" : "Сохранить"}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Быстрые инструменты (показываются только если анализ ещё не запущен) */}
        {!analysisResult && (
          <>
            <div className="card">
              <h2>2) Корреляционный анализ</h2>
              <p className="muted">
                Асинхронный анализ корреляций (работает на любом объёме данных)
              </p>
              <button onClick={onCorrelation} disabled={!file || corrLoading}>
                {corrLoading ? " Загрузка..." : "Запустить корреляцию"}
              </button>

              {corrResult && (
                <div style={{ marginTop: 12 }}>
                  <p>
                    Размер: <b>{corrResult.n_rows}</b> строк ×{" "}
                    <b>{corrResult.n_columns}</b> колонок
                  </p>
                  {renderCorrelationTable()}
                  {corrResult.heatmap && (
                    <div className="plots-grid" style={{ marginTop: 12 }}>
                      <Plot
                        data={corrResult.heatmap.data}
                        layout={corrResult.heatmap.layout}
                        config={{ responsive: true }}
                        style={{ width: "100%" }}
                      />
                    </div>
                  )}
                </div>
              )}

              {corrTaskId && !corrResult && (
                <p className="muted" style={{ marginTop: 8 }}>
                  Вычисление корреляции в фоне...
                </p>
              )}
            </div>

            <div className="card">
              <h2>3) Асинхронные ML-задачи (Celery)</h2>
              <p className="muted">
                Обучение в фоне (требует Celery worker). Не блокирует интерфейс.
              </p>
              <button onClick={onTrainAsync} disabled={!csvData || busy}>
                Обучение модели (Async)
              </button>
              {analysisTaskId && (
                <div className="card" style={{ marginTop: 12 }}>
                  <h3> Статус анализа</h3>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{
                        width: analysisLoading
                          ? "50%"
                          : analysisStatus === "SUCCESS"
                            ? "100%"
                            : "0%",
                      }}
                    />
                  </div>
                  <p>
                    Статус: <b>{analysisStatus}</b>
                    {analysisStatus === "PROGRESS" && analysisResult?.stage && (
                      <> — {analysisResult.stage}</>
                    )}
                  </p>
                  {analysisStatus === "SUCCESS" && !analysisResult && (
                    <p className="ok">Завершено! Загружаются результаты...</p>
                  )}
                </div>
              )}
            </div>

            <PollingTask taskId={trainTaskId} title="Обучение модели" />
          </>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState("checking");

  useEffect(() => {
    healthcheck()
      .then(() => setHealth("ok"))
      .catch(() => setHealth("down"));
  }, []);

  return (
    <HashRouter>
      <ThemeToggle />
      <main className="container">
        <h1>
          Автоматизирования система научных исследований — модуль АРМ
          исследователя
        </h1>
        <p className="muted">React MFE · FastAPI API · ml_core</p>
        <p>
          API:{" "}
          <span
            className={
              health === "ok" ? "ok" : health === "down" ? "error" : ""
            }
          >
            {health}
          </span>
        </p>
        <Tabs />
        <Routes>
          <Route
            path="/"
            element={
              <SafeErrorBoundary fallbackPath="/">
                <MainPage />
              </SafeErrorBoundary>
            }
          />
          <Route
            path="/imputation"
            element={
              <SafeErrorBoundary fallbackPath="/">
                <Imputation />
              </SafeErrorBoundary>
            }
          />
          <Route
            path="/crosstab"
            element={
              <SafeErrorBoundary fallbackPath="/">
                <Crosstab />
              </SafeErrorBoundary>
            }
          />
          <Route
            path="/timeseries"
            element={
              <SafeErrorBoundary fallbackPath="/">
                <TimeSeries />
              </SafeErrorBoundary>
            }
          />
          <Route
            path="/composite"
            element={
              <SafeErrorBoundary fallbackPath="/">
                <CompositeScore />
              </SafeErrorBoundary>
            }
          />

          <Route
            path="/drift"
            element={
              <SafeErrorBoundary fallbackPath="/">
                <DriftCheck />
              </SafeErrorBoundary>
            }
          />
          <Route
            path="/experiments"
            element={
              <SafeErrorBoundary fallbackPath="/">
                <Experiments />
              </SafeErrorBoundary>
            }
          />
          <Route
            path="/combinations"
            element={
              <SafeErrorBoundary fallbackPath="/">
                <FeatureCombinations />
              </SafeErrorBoundary>
            }
          />
          <Route
            path="/subset"
            element={
              <SafeErrorBoundary fallbackPath="/">
                <SubsetSelect />
              </SafeErrorBoundary>
            }
          />
        </Routes>
      </main>
    </HashRouter>
  );
}
