import { useState } from "react";
import { listExperiments, getExperiment } from "../api";

export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [selected, setSelected] = useState("");
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onLoadList() {
    setBusy(true);
    setError("");
    try {
      const res = await listExperiments();
      setExperiments(res.experiments || []);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  async function onLoadDetail() {
    if (!selected) return;
    setBusy(true);
    setError("");
    try {
      const data = await getExperiment(selected);
      console.log("ДАННЫЕ ЭКСПЕРИМЕНТА:", data);
      console.log("data.config:", data?.config);
      console.log("data.config.model_name:", data?.config?.model_name);
      console.log("data.model_name:", data?.model_name);
      setDetail(data);
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  // Форматирование даты
  const formatDate = (timestamp) => {
    if (!timestamp) return "—";
    try {
      const date = new Date(timestamp);
      return date.toLocaleString("ru-RU", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return timestamp;
    }
  };
  console.log("Все эксперименты:", experiments);
  return (
    <div className="card">
      <h2>История экспериментов</h2>
      <p className="muted">
        Чтобы сохранить анализ, на вкладке «Главное» после завершения нажмите
        «Сохранить как эксперимент…» (откроется окно с названием и описанием).
      </p>

      <div style={{ marginTop: 16 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button onClick={onLoadList} disabled={busy}>
            Загрузить список
          </button>
          {busy && <span className="muted">Загрузка...</span>}
        </div>

        {error && <p className="error">{error}</p>}

        {experiments.length > 0 && (
          <>
            <div className="table-wrap" style={{ marginTop: 12 }}>
              <table className="matrix">
                <thead>
                  <tr>
                    <th>Дата</th>
                    <th>Название</th>
                    <th>Модель</th>
                    <th>Цель</th>
                    <th>Признаков</th>
                    <th>F1</th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.map((e, i) => {
                    const modelName =
                      e.config?.model_name || e.model_name || "—";
                    const targetCol =
                      e.config?.target_col || e.target_col || "—";
                    const nFeatures =
                      e.config?.n_features || e.n_features || "—";

                    let f1Value = "—";
                    if (e.metrics?.test?.f1) {
                      f1Value = e.metrics.test.f1.toFixed(3);
                    } else if (e.metrics?.f1) {
                      f1Value = e.metrics.f1.toFixed(3);
                    }

                    return (
                      <tr
                        key={i}
                        style={{ cursor: "pointer" }}
                        onClick={() => setSelected(e.id)}
                      >
                        <td>{formatDate(e.timestamp)}</td>
                        <td>
                          <b>{e.name}</b>
                        </td>
                        <td>{modelName}</td>
                        <td>{targetCol}</td>
                        <td>{nFeatures}</td>
                        <td>{f1Value}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="row" style={{ marginTop: 12, gap: 8 }}>
              <select
                value={selected}
                onChange={(e) => setSelected(e.target.value)}
                style={{ flex: 1 }}
              >
                <option value="">— Выберите эксперимент для деталей —</option>
                {experiments.map((e) => (
                  <option key={e.id} value={e.id}>
                    {formatDate(e.timestamp)} — {e.name}
                  </option>
                ))}
              </select>
              <button onClick={onLoadDetail} disabled={!selected || busy}>
                Загрузить детали
              </button>
            </div>
          </>
        )}

        {experiments.length === 0 && !busy && (
          <p className="muted" style={{ marginTop: 12 }}>
            Нет сохранённых экспериментов. Запустите анализ на главной вкладке и
            нажмите «Сохранить как эксперимент…»
          </p>
        )}

        {/* Детали эксперимента */}
        {detail && (
          <div
            style={{
              marginTop: 20,
              padding: 16,
              background: "var(--bg-secondary)",
              borderRadius: 12,
              border: "1px solid var(--border)",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 16,
              }}
            >
              <h3 style={{ margin: 0 }}>{detail.name}</h3>
              <span className="muted" style={{ fontSize: 12 }}>
                {formatDate(detail.timestamp)}
              </span>
            </div>

            {/* Две колонки: параметры и метрики */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 20,
                marginBottom: 16,
              }}
            >
              {/* Конфигурация анализа */}
              <div>
                <h4 style={{ margin: "0 0 8px 0" }}>Параметры анализа</h4>
                <div
                  style={{
                    background: "var(--bg)",
                    borderRadius: 8,
                    padding: 10,
                  }}
                >
                  {detail.config && Object.keys(detail.config).length > 0 ? (
                    <table style={{ width: "100%", fontSize: 13 }}>
                      <tbody>
                        <tr>
                          <td style={{ padding: "4px 0", fontWeight: 600 }}>
                            Целевая переменная:
                          </td>
                          <td style={{ padding: "4px 0" }}>
                            {detail.config.target_col || "—"}
                          </td>
                        </tr>
                        <tr>
                          <td style={{ padding: "4px 0", fontWeight: 600 }}>
                            Модель:
                          </td>
                          <td style={{ padding: "4px 0" }}>
                            {detail.config?.model_name ||
                              detail.model_name ||
                              "—"}
                          </td>
                        </tr>
                        <tr>
                          <td style={{ padding: "4px 0", fontWeight: 600 }}>
                            SHAP топ-N:
                          </td>
                          <td style={{ padding: "4px 0" }}>
                            {detail.config.shap_top_n || "—"}
                          </td>
                        </tr>
                        <tr>
                          <td style={{ padding: "4px 0", fontWeight: 600 }}>
                            Строк / Признаков:
                          </td>
                          <td style={{ padding: "4px 0" }}>
                            {detail.config.n_samples || detail.n_samples || "—"}{" "}
                            /{" "}
                            {detail.config.n_features ||
                              detail.n_features ||
                              "—"}
                          </td>
                        </tr>
                        <tr>
                          <td style={{ padding: "4px 0", fontWeight: 600 }}>
                            Кластеров / SMOTE:
                          </td>
                          <td style={{ padding: "4px 0" }}>
                            {detail.config?.n_clusters || "—"} /{" "}
                            {detail.config?.use_smote ? "✅" : "❌"}
                          </td>
                        </tr>
                        <tr>
                          <td style={{ padding: "4px 0", fontWeight: 600 }}>
                            Оптимизация гиперпараметров (XGB):
                          </td>
                          <td style={{ padding: "4px 0" }}>
                            {detail.config.use_hp_tuning
                              ? `✅ ${detail.config.n_iter_tuning || 20} итераций`
                              : "❌"}
                          </td>
                        </tr>
                        <tr>
                          <td style={{ padding: "4px 0", fontWeight: 600 }}>
                            Модели:
                          </td>
                          <td style={{ padding: "4px 0" }}>
                            <span
                              style={{
                                color: detail.config.use_lr ? "green" : "gray",
                              }}
                            >
                              LR
                            </span>{" "}
                            <span
                              style={{
                                color: detail.config.use_rf ? "green" : "gray",
                              }}
                            >
                              RF
                            </span>{" "}
                            <span
                              style={{
                                color: detail.config.use_xgb ? "green" : "gray",
                              }}
                            >
                              XGB
                            </span>
                          </td>
                        </tr>
                        <tr>
                          <td style={{ padding: "4px 0", fontWeight: 600 }}>
                            Порог корреляции:
                          </td>
                          <td style={{ padding: "4px 0" }}>
                            {detail.config?.corr_threshold || "—"}
                          </td>
                        </tr>
                        <tr>
                          <td style={{ padding: "4px 0", fontWeight: 600 }}>
                            Метрика оптимизации:
                          </td>
                          <td style={{ padding: "4px 0" }}>
                            {detail.config.optimization_metric
                              ? detail.config.optimization_metric
                              : "F1 (по умолчанию)"}
                          </td>
                        </tr>

                        <tr>
                          <td style={{ padding: "4px 0", fontWeight: 600 }}>
                            Порог риска:
                          </td>
                          <td style={{ padding: "4px 0" }}>
                            {detail.config.risk_threshold}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  ) : (
                    <p className="muted" style={{ margin: 0 }}>
                      Нет данных о конфигурации
                    </p>
                  )}
                </div>
              </div>

              {/* Метрики */}
              <div>
                <h4 style={{ margin: "0 0 8px 0" }}>Метрики модели</h4>
                <div
                  style={{
                    background: "var(--bg)",
                    borderRadius: 8,
                    padding: 10,
                  }}
                >
                  {detail.metrics?.test &&
                  Object.keys(detail.metrics?.test).length > 0 ? (
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(2, 1fr)",
                        gap: 8,
                      }}
                    >
                      {Object.entries(detail.metrics?.test).map(([k, v]) => {
                        if (typeof v === "object" && v !== null) return null;
                        if (k === "cv_results") return null;
                        if (k === "test") return null;
                        return (
                          <div
                            key={k}
                            style={{
                              textAlign: "center",
                              padding: "8px",
                              background: "var(--bg-secondary)",
                              borderRadius: 6,
                            }}
                          >
                            <div
                              style={{
                                fontSize: 11,
                                color: "var(--text-muted)",
                              }}
                            >
                              {k.toUpperCase()}
                            </div>
                            <div style={{ fontSize: 18, fontWeight: "bold" }}>
                              {typeof v === "number" ? v.toFixed(3) : v}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : detail.test_metrics ? (
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(4, 1fr)",
                        gap: 8,
                      }}
                    >
                      {Object.entries(detail.test_metrics).map(([k, v]) => (
                        <div
                          key={k}
                          style={{
                            textAlign: "center",
                            padding: "8px",
                            background: "var(--bg-secondary)",
                            borderRadius: 6,
                          }}
                        >
                          <div
                            style={{ fontSize: 11, color: "var(--text-muted)" }}
                          >
                            {k.toUpperCase()}
                          </div>
                          <div style={{ fontSize: 18, fontWeight: "bold" }}>
                            {typeof v === "number" ? v.toFixed(3) : v}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="muted" style={{ margin: 0 }}>
                      Нет метрик
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Признаки */}
            {detail.features && detail.features.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <h4 style={{ margin: "0 0 6px 0" }}>
                  Признаки ({detail.features.length})
                </h4>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                  {detail.features.map((f, i) => (
                    <span
                      key={i}
                      style={{
                        padding: "4px 10px",
                        background: "var(--primary)",
                        color: "#fff",
                        borderRadius: 16,
                        fontSize: 12,
                      }}
                    >
                      {f}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Кросс-валидация (если есть) */}
            {detail.metrics?.cv_results &&
              Object.keys(detail.metrics.cv_results).length > 0 && (
                <div style={{ marginBottom: 12 }}>
                  <h4 style={{ margin: "0 0 6px 0" }}>Кросс-валидация (F1)</h4>
                  <div className="table-wrap">
                    <table style={{ fontSize: 13 }}>
                      <thead>
                        <tr>
                          <th>Модель</th>
                          <th>Mean ± Std</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(detail.metrics.cv_results).map(
                          ([name, data]) => (
                            <tr key={name}>
                              <td>{name}</td>
                              <td>
                                {data.mean?.toFixed(4)} ± {data.std?.toFixed(4)}
                              </td>
                            </tr>
                          ),
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

            {detail.description && (
              <p
                style={{
                  marginTop: 12,
                  padding: 8,
                  background: "var(--bg)",
                  borderRadius: 6,
                  fontStyle: "italic",
                  fontSize: 13,
                }}
              >
                {detail.description}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
