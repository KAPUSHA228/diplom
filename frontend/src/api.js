const API_BASE = import.meta.env.VITE_API_BASE || "";
async function request(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000); // 30 сек
  try {
    const res = await fetch(`${API_BASE}${url}`, { ...options, signal: controller.signal });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }
    return res.json();
  } finally {
    clearTimeout(timeout);
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
      body: form
    });
}
export async function getTaskStatus(taskId) {
    return request(`/api/v1/ml/tasks/${taskId}`);
}

export async function uploadForCorrelation(file) {
    const form = new FormData();
    form.append("file", file);
    return request("/api/v1/ml/correlation", { method: "POST", body: form });
}

// ==================== Новые эндпоинты (endpoints.py) ====================

/** Полный анализ через ResearchAnalyzer */
export async function runFullAnalysis(data) {
    return request("/api/v1/analyze/full", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ df: data }),
    });
}

/** Создание композитной оценки */
export async function createCompositeScore(data, featureWeights, scoreName = "custom_score") {
    return request("/api/v1/analyze/composite/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ df: data, feature_weights: featureWeights, score_name: scoreName }),
    });
}

/** Выделение подмножества респондентов */
export async function selectSubset(data, condition = null, nSamples = null, byCluster = null) {
    return request("/api/v1/analyze/subset/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ df: data, condition, n_samples: nSamples, by_cluster: byCluster }),
    });
}

// ==================== Импутация ====================

/** Обработка пропусков и выбросов */
export async function handleImputation(data, strategy = "auto", threshold = 30.0) {
    return request("/api/v1/analyze/imputation/handle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ df: data, strategy, threshold }),
    });
}

// ==================== Кросс-таблицы ====================

/** Полная кросс-таблица с χ²-тестом */
export async function buildCrosstab(data, rowVar, colVar, values = null, aggfunc = "count") {
    return request("/api/v1/analyze/crosstab", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ df: data, row_var: rowVar, col_var: colVar, values, aggfunc }),
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
export async function getTrajectory(data, studentId, valueCol = "avg_grade", timeCol = "semester") {
    return request("/api/v1/analyze/timeseries/trajectory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ df: data, student_id: studentId, value_col: valueCol, time_col: timeCol }),
    });
}

/** Поиск негативной динамики */
export async function findNegativeDynamics(data, valueCol = "avg_grade", timeCol = "semester") {
    return request("/api/v1/analyze/timeseries/negative_dynamics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ df: data, value_col: valueCol, time_col: timeCol }),
    });
}

/** Прогноз оценок */
export async function forecastStudent(data, studentId, valueCol = "avg_grade", timeCol = "semester", futureSemesters = 2) {
    return request("/api/v1/analyze/timeseries/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ df: data, student_id: studentId, value_col: valueCol, time_col: timeCol, future_semesters: futureSemesters }),
    });
}

// ==================== Дрейф ====================

/** Проверка дрейфа данных */
export async function checkDrift(referenceData, currentData, modelName = "unknown") {
    return request("/api/v1/analyze/drift/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reference_data: referenceData, current_data: currentData, model_name: modelName }),
    });
}

// ==================== Эксперименты ====================

/** Сохранение эксперимента */
export async function saveExperiment(name, metrics = {}, features = [], description = "") {
    return request("/api/v1/analyze/experiments/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, metrics, features, description }),
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
