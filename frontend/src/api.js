const API_BASE = import.meta.env.VITE_API_BASE || "";

const pendingRequests = new Map();

async function request(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300000);

  // Только для НЕ polling запросов
  const cancelKey = options.cancelKey || (options.noCancel ? null : url);

  if (cancelKey && pendingRequests.has(cancelKey)) {
    pendingRequests.get(cancelKey).abort();
  }
  if (cancelKey) {
    pendingRequests.set(cancelKey, controller);
  }

  try {
    const res = await fetch(`${API_BASE}${url}`, {
      ...options,
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    return res.json();
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("Request was cancelled");
    }
    throw err;
  } finally {
    clearTimeout(timeout);
    if (cancelKey) {
      pendingRequests.delete(cancelKey);
    }
  }
}

//class WebSocketManager {
//  constructor() {
//    this.connections = new Map(); // taskId -> { ws, listeners }
//    this.reconnectAttempts = new Map();
//    this.maxReconnectAttempts = 3;
//  }
//
//  connect(taskId, onMessage, onError, onClose) {
//    // Если уже есть соединение для этого taskId — не создаём новое
//    if (this.connections.has(taskId)) {
//      const existing = this.connections.get(taskId);
//      existing.listeners.push({ onMessage, onError, onClose });
//      return () => this.disconnect(taskId, existing.listeners.length - 1);
//    }
//
//    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
//    const wsUrl = `${protocol}//${window.location.host}/api/v1/ml/ws/task/${taskId}`;
//    const ws = new WebSocket(wsUrl);
//
//    const listeners = [{ onMessage, onError, onClose }];
//
//    ws.onopen = () => {
//      console.log(`[WebSocket] Connected to task ${taskId}`);
//      this.reconnectAttempts.delete(taskId);
//    };
//
//    ws.onmessage = (event) => {
//      const data = JSON.parse(event.data);
//      listeners.forEach((l) => l.onMessage?.(data));
//    };
//
//    ws.onerror = (err) => {
//      console.error(`[WebSocket] Error for task ${taskId}:`, err);
//      listeners.forEach((l) => l.onError?.(err));
//    };
//
//    ws.onclose = () => {
//      console.log(`[WebSocket] Closed for task ${taskId}`);
//
//      // Попытка переподключения
//      const attempts = this.reconnectAttempts.get(taskId) || 0;
//      if (attempts < this.maxReconnectAttempts) {
//        this.reconnectAttempts.set(taskId, attempts + 1);
//        setTimeout(
//          () => {
//            if (this.connections.has(taskId)) {
//              this.connect(taskId, onMessage, onError, onClose);
//            }
//          },
//          1000 * (attempts + 1),
//        );
//      } else {
//        listeners.forEach((l) => l.onClose?.());
//        this.connections.delete(taskId);
//        this.reconnectAttempts.delete(taskId);
//      }
//    };
//
//    this.connections.set(taskId, { ws, listeners });
//
//    // Возвращаем функцию отписки
//    return () => this.disconnect(taskId, listeners.length - 1);
//  }
//
//  disconnect(taskId, listenerIndex = null) {
//    const conn = this.connections.get(taskId);
//    if (!conn) return;
//
//    if (listenerIndex !== null) {
//      // Удаляем только одного слушателя
//      conn.listeners.splice(listenerIndex, 1);
//      if (conn.listeners.length > 0) return;
//    }
//
//    // Закрываем соединение, если слушателей не осталось
//    conn.ws.close();
//    this.connections.delete(taskId);
//    this.reconnectAttempts.delete(taskId);
//  }
//
//  send(taskId, data) {
//    const conn = this.connections.get(taskId);
//    if (conn && conn.ws.readyState === WebSocket.OPEN) {
//      conn.ws.send(JSON.stringify(data));
//    }
//  }
//}
//
//export const wsManager = new WebSocketManager();
//
//class PollingClient {
//    constructor() {
//        this.activePolling = new Map();
//    }
//
//    poll(taskId, onUpdate, options = {}) {
//        const { intervalMs = 3000, maxAttempts = 200 } = options;
//
//        // Если уже есть активный polling для этого taskId — не создаём новый
//        if (this.activePolling.has(taskId)) {
//            console.warn(`[PollingClient] Already polling task ${taskId}`);
//            // ВОЗВРАЩАЕМ ФУНКЦИЮ, А НЕ undefined
//            return () => {};
//        }
//
//        let attempts = 0;
//        let isActive = true;
//        let timeoutId = null;
//
//        const stop = () => {
//            console.log(`[PollingClient] Stopping polling for task ${taskId}`);
//            isActive = false;
//            if (timeoutId) {
//                clearTimeout(timeoutId);
//                timeoutId = null;
//            }
//            this.activePolling.delete(taskId);
//        };
//
//        this.activePolling.set(taskId, { stop });
//
//        const pollOnce = async () => {
//            if (!isActive) return;
//
//            try {
//                const response = await fetch(`${API_BASE}/api/v1/ml/full_async/${taskId}`);
//                const data = await response.json();
//
//                if (!isActive) return;
//
//                if (onUpdate) {
//                    onUpdate(data);
//                }
//                attempts++;
//
//                if (data.status === 'SUCCESS' || data.status === 'FAILURE' || attempts >= maxAttempts) {
//                    stop();
//                    return;
//                }
//
//                const delay = Math.min(intervalMs * Math.pow(1.2, Math.floor(attempts / 10)), 15000);
//                timeoutId = setTimeout(pollOnce, delay);
//
//            } catch (err) {
//                console.error(`[PollingClient] Error polling task ${taskId}:`, err);
//                if (isActive) {
//                    timeoutId = setTimeout(pollOnce, 5000);
//                }
//            }
//        };
//
//        pollOnce();
//        return stop;  // ← ВАЖНО: возвращаем функцию stop
//    }
//
//    stop(taskId) {
//        const polling = this.activePolling.get(taskId);
//        if (polling && polling.stop) {
//            polling.stop();
//        }
//    }
//}
//
//export const pollingClient = new PollingClient();

// Специальная функция для отмены всех запросов к задаче
export function cancelTaskPolling(taskId) {
  const key = `/api/v1/ml/full_async/${taskId}`;
  if (pendingRequests.has(key)) {
    pendingRequests.get(key).abort();
    pendingRequests.delete(key);
  }
}

export async function healthcheck() {
  return request("/health");
}

export async function uploadForTrain(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/api/v1/ml/train", { method: "POST", body: form });
}

export async function uploadForShap(file, modelId = "XGB") {
  const form = new FormData();
  form.append("file", file);
  return request(`/api/v1/ml/shap?model_id=${encodeURIComponent(modelId)}`, {
    method: "POST",
    body: form,
  });
}
/** Получение статуса задачи */
export async function getTaskStatus(taskId) {
  return request(`/api/v1/ml/tasks/${taskId}`);
}

export async function cancelFullAnalysis(taskId) {
  return request(`/api/v1/ml/full_async/${taskId}/cancel`, { method: "POST" });
}

export async function processCSV(
  file,
  sheetGroup = "numeric",
  mappingConfig = null,
) {
  const form = new FormData();
  form.append("file", file);
  form.append("sheet_group", sheetGroup);
  if (mappingConfig) {
    form.append("mapping_config", JSON.stringify(mappingConfig));
  }
  return request("/api/v1/analyze/csv/process", {
    method: "POST",
    body: form,
  });
}
export async function runCorrelationAsync(data, targetCol = null) {
  return request("/api/v1/ml/correlation_async", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ df: data, target_col: targetCol }),
  });
}
export async function getCorrelationStatus(taskId) {
  return request(`/api/v1/ml/correlation_async/${taskId}`);
}
// ==================== Новые эндпоинты (endpoints.py) ====================

/** Полный анализ через ResearchAnalyzer */
export async function runFullAnalysis(data, params = {}) {
  return request("/api/v1/analyze/full", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      df: data,
      target_col: params.target_col || "risk_flag",
      n_clusters: params.n_clusters || 3,
      corr_threshold: params.corr_threshold || 0.3,
      risk_threshold: params.risk_threshold || 0.5,
      n_iter_tuning: params.n_iter_tuning || 10,
      use_smote: params.use_smote !== undefined ? params.use_smote : true,
      // Параметры из сайдбара
      use_hp_tuning: params.use_hp_tuning || false,
      optimization_metric:
        params.optimization_metric !== "default"
          ? params.optimization_metric
          : null,
      n_features_to_select: params.n_features_to_select || 7,
      shap_top_n: params.shap_top_n || 5,
      use_lr: params.use_lr !== undefined ? params.use_lr : true,
      use_rf: params.use_rf !== undefined ? params.use_rf : true,
      use_xgb: params.use_xgb !== undefined ? params.use_xgb : true,
    }),
  });
}

export async function savePlotToFile(
  figure,
  filename = "plot",
  format = "png",
) {
  const response = await fetch("/api/v1/analyze/plot/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ figure, filename, format }),
  });

  if (!response.ok) throw new Error("Failed to save plot");

  // Получаем файл и скачиваем
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${filename}.${format}`;
  link.click();
  URL.revokeObjectURL(url);
}

/** Создание композитной оценки */
export async function createCompositeScore(
  data,
  featureWeights,
  scoreName = "custom_score",
) {
  return request("/api/v1/analyze/composite/create", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      df: data,
      feature_weights: featureWeights,
      score_name: scoreName,
    }),
  });
}

/** Выделение подмножества респондентов */
export async function selectSubset(
  data,
  condition = null,
  nSamples = null,
  byCluster = null,
  randomSeed = null,
) {
  return request("/api/v1/analyze/subset/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      df: data,
      condition,
      n_samples: nSamples,
      by_cluster: byCluster,
      random_seed: randomSeed,
    }),
  });
}

/** Создание комбинированных признаков */
export async function createFeatureCombinations(
  data,
  numericalCols,
  textCols,
  maxPairs = 15,
  targetCol = null,
) {
  return request("/api/v1/analyze/combinations/create", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      df: data,
      numerical_cols: numericalCols,
      text_cols: textCols,
      max_pairs: maxPairs,
      target_col: targetCol,
    }),
  });
}

// ==================== Импутация ====================

/** Обработка пропусков и выбросов */
export async function handleImputation(
  data,
  strategy = "auto",
  threshold = 30.0,
) {
  return request("/api/v1/analyze/imputation/handle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ df: data, strategy, threshold }),
  });
}

// ==================== Кросс-таблицы ====================

/** Полная кросс-таблица с χ²-тестом */
export async function buildCrosstab(
  data,
  rowVar,
  colVar,
  values = null,
  aggfunc = "count",
  nBins = 4,
  binMethod = "cut",
) {
  return request("/api/v1/analyze/crosstab", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      df: data,
      row_var: rowVar,
      col_var: colVar,
      values: values,
      aggfunc: aggfunc,
      n_bins: nBins,
      bin_method: binMethod,
    }),
  });
}

/** Упрощённая кросс-таблица */
export async function buildSimpleCrosstab(data, rowVar, colVar) {
  return request("/api/v1/analyze/crosstab/simple", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ df: data, row_var: rowVar, col_var: colVar }),
  });
}

// ==================== Временные ряды ====================

/** Траектория студента */
export async function getTrajectory(
  data,
  studentId,
  valueCol = "avg_grade",
  timeCol = "semester",
) {
  return request("/api/v1/analyze/timeseries/trajectory", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      df: data,
      student_id: studentId,
      value_col: valueCol,
      time_col: timeCol,
    }),
  });
}

/** Поиск негативной динамики */
export async function findNegativeDynamics(
  data,
  valueCol = "avg_grade",
  timeCol = "semester",
) {
  return request("/api/v1/analyze/timeseries/negative_dynamics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ df: data, value_col: valueCol, time_col: timeCol }),
  });
}

/** Прогноз оценок */
export async function forecastStudent(
  data,
  studentId,
  valueCol = "avg_grade",
  timeCol = "semester",
  futureSemesters = 2,
) {
  return request("/api/v1/analyze/timeseries/forecast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      df: data,
      student_id: studentId,
      value_col: valueCol,
      time_col: timeCol,
      future_semesters: futureSemesters,
    }),
  });
}

// ==================== Асинхронные задачи (Celery) ====================

/** Запуск обучения в фоне (принимает JSON) */
export async function trainAsyncJson(data, targetCol = "risk_flag") {
  return request("/api/v1/ml/train_async", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ df: data, target_col: targetCol }),
  });
}

export async function runFullAnalysisAsync(data, params = {}) {
  console.log("🔵 [API] Получены params:", params);
  console.log("🔵 [API] use_hp_tuning =", params.use_hp_tuning);
  console.log("🔵 [API] n_iter_tuning =", params.n_iter_tuning);
  const body = {
    df: data,
    target_col: params.target_col || "risk_flag",
    n_clusters: params.n_clusters || 3,
    risk_threshold: params.risk_threshold || 0.5,
    corr_threshold: params.corr_threshold || 0.3,
    use_smote: params.use_smote !== undefined ? params.use_smote : true,
    n_features_to_select: params.n_features_to_select || 7,
    is_synthetic: params.is_synthetic || false,
    use_hp_tuning: params.use_hp_tuning === true,
    n_iter_tuning: params.n_iter_tuning || 20,
    optimization_metric:
      params.optimization_metric !== "default"
        ? params.optimization_metric
        : null,
    shap_top_n: params.shap_top_n || 5,
    use_lr: params.use_lr !== undefined ? params.use_lr : true,
    use_rf: params.use_rf !== undefined ? params.use_rf : true,
    use_xgb: params.use_xgb !== undefined ? params.use_xgb : true,
  };
  console.log("🔵 [API] Отправляем JSON в бэкенд:", body);
  return request("/api/v1/ml/full_async", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function getFullAnalysisStatus(taskId) {
  return request(`/api/v1/ml/full_async/${taskId}`);
}

// ==================== Дрейф ====================

/** Проверка дрейфа данных */
export async function checkDrift(
  referenceData,
  currentData,
  modelName = "unknown",
) {
  return request("/api/v1/analyze/drift/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      reference_data: referenceData,
      current_data: currentData,
      model_name: modelName,
    }),
  });
}

/** История метрик моделей */
export async function getMetricsHistory() {
  return request("/api/v1/analyze/metrics/history");
}

// ==================== Эксперименты ====================

/** Сохранение эксперимента */
export async function saveExperiment(
  name,
  metrics = {},
  features = [],
  description = "",
  config = {},
) {
  return request("/api/v1/analyze/experiments/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      metrics,
      features,
      description,
      config, // <--- Отправляем конфигурацию для воспроизводимости
    }),
  });
}

/** Список экспериментов */
export async function listExperiments(limit = 20) {
  return request(`/api/v1/analyze/experiments/list?limit=${limit}`);
}

/** Загрузка эксперимента по ID */
export async function getExperiment(experimentId) {
  return request(`/api/v1/analyze/experiments/${experimentId}`);
}

// ==================== Excel Mapping ====================

/** Получить превью листа (типы колонок и уникальные значения) */
export async function getExcelPreview(file, sheetName = "0") {
  const form = new FormData();
  form.append("file", file);
  form.append("sheet_name", sheetName);
  console.log("[api.js] getExcelPreview: отправляю sheet_name =", sheetName);
  return request("/api/v1/analyze/excel/preview", {
    method: "POST",
    body: form,
  });
}

/** Обработать лист Excel с настройками маппинга */
export async function processExcel(
  file,
  sheetName = "0",
  sheetGroup = "numeric",
  mappingConfig = null,
) {
  const form = new FormData();
  form.append("file", file);
  form.append("sheet_name", sheetName);
  form.append("sheet_group", sheetGroup);
  if (mappingConfig) {
    form.append("mapping_config", JSON.stringify(mappingConfig));
  }
  return request("/api/v1/analyze/excel/process", {
    method: "POST",
    body: form,
  });
}
