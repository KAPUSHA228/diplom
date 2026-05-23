import { useState, useEffect, useMemo } from "react";
import Plot from "react-plotly.js";
import { getTrajectory, findNegativeDynamics, forecastStudent } from "../api";
import { useDatasetLoader } from "../hooks/useDatasetLoader";

export default function TimeSeries() {
  const {
    data: sharedData,
    columns: sharedCols,
    hasData,
    loading,
  } = useDatasetLoader();

  // Режимы анализа
  const [mode, setMode] = useState("individual");
  const [valueCol, setValueCol] = useState("");
  const [timeCol, setTimeCol] = useState("");
  const [selectedStudent, setSelectedStudent] = useState("");

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // Защита от undefined — безопасные массивы
  const safeData = useMemo(() => sharedData || [], [sharedData]);
  const safeCols = useMemo(() => sharedCols || [], [sharedCols]);

  console.log("[TimeSeries] sharedData length:", safeData.length);
  console.log("[TimeSeries] columns:", safeCols);
  console.log(
    "[TimeSeries] students found:",
    [...new Set(safeData.map((d) => d.student_id ?? d.user_id).filter(Boolean))]
      .length,
  );

  // Доступные опции
  const numericCols = useMemo(() => {
    return safeCols.filter((col) => {
      if (!safeData.length) return false;
      const firstRow = safeData[0];
      return firstRow && typeof firstRow[col] === "number";
    });
  }, [safeCols, safeData]);

  const students = useMemo(() => {
    return [
      ...new Set(
        safeData
          .map((d) => d.student_id ?? d.user_id ?? d.StudentID)
          .filter(Boolean),
      ),
    ];
  }, [safeData]);

  // Авто-выбор первого числового показателя
  useEffect(() => {
    if (numericCols.length > 0 && !valueCol) {
      setValueCol(numericCols[0]);
    }
  }, [numericCols, valueCol]);

  // Авто-выбор временной колонки (ищем semester или date)
  useEffect(() => {
    if (!timeCol && safeCols.length) {
      const timeCandidate = safeCols.find(
        (c) =>
          c.toLowerCase().includes("semester") ||
          c.toLowerCase().includes("date") ||
          c.toLowerCase().includes("time"),
      );
      if (timeCandidate) setTimeCol(timeCandidate);
      else if (safeCols[0]) setTimeCol(safeCols[0]);
    }
  }, [safeCols, timeCol]);

  if (loading) {
    return <div className="card">Загрузка данных...</div>;
  }
  // Показываем индикатор загрузки
  if (loading) {
    return (
      <div className="card">
        <p>Загрузка данных...</p>
      </div>
    );
  }

  // Если нет данных
  if (!hasData) {
    return (
      <div className="card">
        <p className="muted">Нет данных. Загрузите файл на главной странице.</p>
      </div>
    );
  }

  async function runAnalysis() {
    if (!safeData.length) {
      setError("Нет данных для анализа");
      return;
    }
    setBusy(true);
    setError("");
    setResult(null);

    try {
      console.log(`[TimeSeries] Запуск ${mode} анализа`, {
        valueCol,
        timeCol,
        selectedStudent,
      });

      if (mode === "individual" && selectedStudent) {
        const res = await getTrajectory(
          safeData,
          selectedStudent,
          valueCol,
          timeCol,
        );
        console.log("[TimeSeries] Trajectory result:", res);
        console.log("[TimeSeries] Individual result:", res);
        console.log(
          "[TimeSeries] FULL API response:",
          JSON.stringify(res, null, 2),
        );
        console.log("[TimeSeries] Response has chart?", !!res.chart);
        console.log("[TimeSeries] Response has figure?", !!res.figure);
        console.log("[TimeSeries] Response keys:", Object.keys(res));
        // =========================================
        setResult({ mode, data: res });
      } else if (mode === "negative") {
        const res = await findNegativeDynamics(safeData, valueCol, timeCol);
        console.log("[TimeSeries] Negative dynamics result:", res);
        setResult({ mode, data: res });
      } else if (mode === "forecast" && selectedStudent) {
        const res = await forecastStudent(
          safeData,
          selectedStudent,
          valueCol,
          timeCol,
          3,
        );
        console.log("[TimeSeries] Forecast result:", res);
        setResult({ mode, data: res });
      }
    } catch (e) {
      console.error(e);
      setError(e.message || "Ошибка выполнения");
    } finally {
      setBusy(false);
    }
  }

  if (hasData && students.length > 0) {
    const avgRowsPerStudent = safeData.length / students.length;
    if (avgRowsPerStudent < 2.5) {
      return (
        <div className="card">
          <h2>Временные ряды и траектории студентов</h2>
          <div
            className="warning"
            style={{ padding: 16, background: "#000000", borderRadius: 8 }}
          >
            <h3>⚠️Данные не подходят для анализа временных рядов</h3>
            <p>
              Текущий датасет содержит{" "}
              <b>примерно {avgRowsPerStudent.toFixed(1)} записей на студента</b>
              .<br />
              Для анализа траекторий, негативной динамики и прогнозов необходимо
              иметь
              <b>несколько наблюдений на одного студента</b> (минимум 3–4
              семестра).
            </p>
            <p className="muted">
              Рекомендация: загрузите данные в формате панельных (long-format),
              где у каждого студента несколько строк с разными значениями
              времени.
            </p>
          </div>
        </div>
      );
    }
  }
  return (
    <div className="card">
      <h2>Временные ряды и траектории студентов</h2>

      {hasData && (
        <div className="ok" style={{ marginBottom: 16 }}>
          Анализируется <b>{safeData.length}</b> записей • Числовых показателей:{" "}
          <b>{numericCols.length}</b>
        </div>
      )}

      {/* Выбор режима */}
      <div
        className="mode-selector"
        style={{ marginBottom: 20, display: "flex", gap: 8 }}
      >
        <button
          className={mode === "individual" ? "active" : ""}
          onClick={() => setMode("individual")}
        >
          Индивидуальная траектория
        </button>
        <button
          className={mode === "negative" ? "active" : ""}
          onClick={() => setMode("negative")}
        >
          Негативная динамика
        </button>
        <button
          className={mode === "forecast" ? "active" : ""}
          onClick={() => setMode("forecast")}
        >
          Прогнозирование
        </button>
      </div>

      <div
        className="row"
        style={{ gap: 12, marginBottom: 16, alignItems: "end" }}
      >
        <div>
          <label>Показатель:</label>
          <select
            value={valueCol}
            onChange={(e) => setValueCol(e.target.value)}
          >
            {numericCols.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label>Временная ось:</label>
          <select value={timeCol} onChange={(e) => setTimeCol(e.target.value)}>
            {safeCols.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>

        {(mode === "individual" || mode === "forecast") && (
          <div>
            <label>Студент:</label>
            <select
              value={selectedStudent}
              onChange={(e) => setSelectedStudent(e.target.value)}
              style={{ minWidth: 180 }}
            >
              <option value="">— Выберите студента —</option>
              {students.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <button
        onClick={runAnalysis}
        disabled={busy || (mode !== "negative" && !selectedStudent)}
        className="primary"
      >
        {busy ? "Выполняется анализ..." : "Запустить анализ"}
      </button>

      {error && <p className="error">{error}</p>}

      {/* === РЕЗУЛЬТАТЫ === */}
      {result && (
        <div style={{ marginTop: 24 }}>
          {result.mode === "individual" &&
            result.data?.chart &&
            selectedStudent && (
              <>
                <h3>Траектория студента {selectedStudent}</h3>
                <Plot
                  data={result.data.chart.data}
                  layout={result.data.chart.layout}
                  config={{ responsive: true }}
                  style={{ width: "100%" }}
                />
              </>
            )}

          {result.mode === "negative" && result.data && (
            <>
              <h3>Студенты с негативной динамикой</h3>
              <p>
                Найдено <b>{result.data.at_risk_count || 0}</b> студентов из{" "}
                <b>{result.data.n_students_analyzed || 0}</b> (
                {(result.data.risk_percentage || 0).toFixed(1)}%)
              </p>
              {result.data.at_risk_students?.length > 0 && (
                <div className="table-wrap">
                  <table className="risk-table">
                    <thead>
                      <tr>
                        <th>ID студента</th>
                        <th>Начальное значение</th>
                        <th>Конечное значение</th>
                        <th>Изменение</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.data.at_risk_students.map((student, idx) => (
                        <tr key={idx}>
                          <td>
                            {student.student_id || student.user_id || idx}
                          </td>
                          <td>{student.first_value?.toFixed(2)}</td>
                          <td>{student.last_value?.toFixed(2)}</td>
                          <td className="error">
                            {(
                              ((student.last_value - student.first_value) /
                                student.first_value) *
                              100
                            ).toFixed(1)}
                            %
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}

          {result.mode === "forecast" &&
            result.data.predictions &&
            selectedStudent && (
              <>
                <h3>Прогноз для студента {selectedStudent}</h3>
                <p>
                  Тренд: <b>{result.data.trend?.toFixed(3) || "—"}</b>
                </p>
                {result.data.chart && (
                  <Plot
                    data={result.data.chart.data}
                    layout={result.data.chart.layout}
                    config={{ responsive: true }}
                    style={{ width: "100%" }}
                  />
                )}
              </>
            )}
        </div>
      )}
    </div>
  );
}
