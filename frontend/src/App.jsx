import { useEffect, useMemo, useState } from "react";
import { HashRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import Plot from "react-plotly.js";
import * as XLSX from "xlsx";
import {
  healthcheck,
  uploadForCorrelation,
  runFullAnalysis,
  getExcelPreview,
  processExcel,
} from "./api";
import AnalysisSidebar from "./components/AnalysisSidebar";
import AnalysisResults from "./components/AnalysisResults";
import SheetMapper from "./components/SheetMapper";
import DataEnrichment from "./components/DataEnrichment";
import Imputation from "./components/Imputation";
import DriftCheck from "./components/DriftCheck";
import Crosstab from "./components/Crosstab";
import TimeSeries from "./components/TimeSeries";
import CompositeScore from "./components/CompositeScore";
import Experiments from "./components/Experiments";
import SubsetSelect from "./components/SubsetSelect";
import FeatureCombinations from "./components/FeatureCombinations";
import { handleImputation } from "./api";
import { useSharedData } from "./hooks/useSharedData";
import "./styles.css";

const NAV = [
  { path: "/", label: "📊 Главное" },
  { path: "/imputation", label: "🔧 Пропуски" },
  { path: "/crosstab", label: "📈 Кросс-таблицы" },
  { path: "/timeseries", label: "📉 Временные ряды" },
  { path: "/composite", label: "🎯 Композитные оценки" },
  { path: "/combinations", label: "🔗 Комбинации" },
  { path: "/drift", label: "🔄 Дрейф" },
  { path: "/experiments", label: "📁 Эксперименты" },
  { path: "/subset", label: "📋 Подмножество" },
];

function Tabs() {
  const location = useLocation();
  return (
    <div className="tabs">
      {NAV.map((n) => (
        <Link key={n.path} to={n.path} className={location.pathname === n.path ? "active" : ""}>
          {n.label}
        </Link>
      ))}
    </div>
  );
}

/** Кнопка переключения темы */
function ThemeToggle() {
  const [dark, setDark] = useState(() => !document.documentElement.classList.contains("light"));

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

function MainPage() {
  const shared = useSharedData();
  const [file, setFile] = useState(null);
  const [csvData, setCsvData] = useState(null);
  const [csvPreview, setCsvPreview] = useState({ headers: [], rows: [], rowCount: 0, riskPct: null });
  const [corrResult, setCorrResult] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  // Восстановление из shared dataset при монтировании (переключение вкладок / F5)
  useEffect(() => {
    console.log("[MainPage] shared:", { hasShared: shared.hasShared, dataLen: shared.data?.length, csvData: !!csvData, file: !!file });
    if (shared.hasShared && !csvData && !file) {
      setCsvData(shared.data);
      if (shared.data.length > 0) {
        const headers = Object.keys(shared.data[0]);
        const previewRows = shared.data.slice(0, 5).map(r => headers.map(h => String(r[h] ?? "")));
        setCsvPreview({ headers, rows: previewRows, rowCount: shared.data.length });
        console.log("[MainPage] данные восстановлены:", shared.data.length, "строк");
      }
    }
  }, []);

  // Excel Mapping state
  const [sheetPreview, setSheetPreview] = useState(null); // Данные для SheetMapper
  const [mappingConfig, setMappingConfig] = useState(null); // Конфиг для обработки
  const [excelSheets, setExcelSheets] = useState([]); // Имена листов в файле
  const [selectedSheet, setSelectedSheet] = useState(""); // Выбранный лист
  const [rawExcelData, setRawExcelData] = useState(null); // Сырые данные после processExcel (до обогащения)

  // Выбор целевой переменной (как в Streamlit)
  const [targetColumn, setTargetColumn] = useState("");
  const [targetSelected, setTargetSelected] = useState(false);

  // Служебные колонки — исключаем из выбора target И из превью
  const EXCLUDE_COLS = new Set([
    "user", "user_id", "vk_id", "vk id", "vk",
    "фамилия", "имя", "отчество",
    "вуз", "факультет", "группа", "курс",
    "пол", "возраст",
    "дата", "date",
    "направление подготовки",
    "_source_sheet", "_sheet_type",
  ]);

  // Фильтрация служебных колонок из клиентских данных
  function filterServiceCols(data) {
    if (!data || !data.length) return data;
    return data.map(row => {
      const filtered = {};
      for (const [key, val] of Object.entries(row)) {
        if (!EXCLUDE_COLS.has(key.toLowerCase())) {
          filtered[key] = val;
        }
      }
      return filtered;
    });
  }

  // Доступные колонки для выбора target
  const targetCandidates = useMemo(() => {
    if (!csvData || !csvData.length) return [];
    const allKeys = Object.keys(csvData[0]);
    return allKeys.filter((k) => !EXCLUDE_COLS.has(k.toLowerCase()));
  }, [csvData]);

  // Статистика по выбранной колонке
  const targetStats = useMemo(() => {
    if (!targetColumn || !csvData) return null;
    const vals = csvData.map((r) => r[targetColumn]).filter((v) => v !== "" && v != null);
    const unique = new Set(vals).size;
    const isNumeric = vals.every((v) => typeof v === "number");
    return {
      type: isNumeric ? "числовой" : "категориальный",
      unique,
      min: isNumeric ? Math.min(...vals).toFixed(2) : null,
      max: isNumeric ? Math.max(...vals).toFixed(2) : null,
      needsBinarize: isNumeric && unique > 2,
    };
  }, [targetColumn, csvData]);

  const canRun = useMemo(() => !!csvData && targetSelected && !busy, [csvData, targetSelected, busy]);

  async function onRunAnalysis(params) {
    if (!csvData) return;
    setBusy(true); setError(""); setAnalysisResult(null);
    try {
      const res = await runFullAnalysis(csvData, { ...params, target_col: targetColumn });
      setAnalysisResult(res);
      // Сохраняем в localStorage для Experiments.jsx
      localStorage.setItem("last_analysis_result", JSON.stringify(res));
    } catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  async function onCorrelation() {
    if (!file) return;
    try { setBusy(true); setError(""); const res = await uploadForCorrelation(file); setCorrResult(res); }
    catch (e) { setError(String(e.message || e)); }
    finally { setBusy(false); }
  }

  /** Вспомогательная функция для обновления превью и данных */
  function setDataAndPreview(data) {
    const filtered = filterServiceCols(data);
    setCsvData(filtered);
    shared.updateData(filtered);
    if (filtered && filtered.length > 0) {
      const headers = Object.keys(filtered[0]);
      const previewRows = filtered.slice(0, 5).map(r => headers.map(h => String(r[h] ?? "")));
      setCsvPreview({ headers, rows: previewRows, rowCount: filtered.length });
    } else {
      setCsvPreview({ headers: [], rows: [], rowCount: 0, riskPct: null });
    }
  }

  /** Обработчик подтверждения маппинга из SheetMapper */
  async function onMappingConfirm(config) {
    if (!file) return;
    setBusy(true);
    setMappingConfig(config);
    try {
      const group = sheetPreview?.detected_group || "numeric";
      const res = await processExcel(file, sheetPreview?.sheet_name || "0", group, config);
      // Сохраняем сырые данные и показываем DataEnrichment
      setRawExcelData(res.data);
      // sheetPreview НЕ сбрасываем — он нужен для отображения категории в DataEnrichment
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
      setDataAndPreview(res.data);
      setRawExcelData(null);
      setSheetPreview(null); // Больше не нужен
      setTargetSelected(false);
    } catch (e) {
      setError("Ошибка обогащения: " + e.message);
    } finally {
      setBusy(false);
    }
  }

  /** Обработчик пропуска обогащения */
  function onEnrichmentSkip() {
    if (!rawExcelData) return;
    setDataAndPreview(rawExcelData);
    setRawExcelData(null);
    setSheetPreview(null); // Больше не нужен
    setTargetSelected(false);
  }

  /** Загружает превью выбранного листа Excel */
  async function loadSheetPreview(fileObj, sheetName) {
    if (!fileObj || !sheetName) return;
    setBusy(true); setError("");
    try {
      console.log("=== loadSheetPreview ===");
      console.log("Выбранный лист:", sheetName);
      const preview = await getExcelPreview(fileObj, sheetName);
      console.log("Превью от сервера:", preview);
      setSheetPreview(preview);
      setBusy(false);

      // ВСЕ листы (и числовые тоже) теперь идут через процесс → DataEnrichment
      const group = preview.detected_group || "numeric";
      const res = await processExcel(fileObj, sheetName, group, null);
      setRawExcelData(res.data);
    } catch (e) {
      setError("Ошибка превью: " + e.message);
      setBusy(false);
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

  /** Обрабатывает загрузку CSV или Excel */
  async function onFileChange(e) {
    const next = e.target.files?.[0] || null;
    setFile(next); setCorrResult(null); setAnalysisResult(null); setCsvData(null);
    setSheetPreview(null); setMappingConfig(null); setRawExcelData(null);
    setExcelSheets([]); setSelectedSheet("");
    setTargetColumn(""); setTargetSelected(false);
    if (!next) { setCsvPreview({ headers: [], rows: [], rowCount: 0, riskPct: null }); return; }

    const isExcel = next.name.endsWith(".xlsx") || next.name.endsWith(".xls");

    try {
      if (isExcel) {
        // --- Excel: читаем имена листов локально ---
        const buf = await next.arrayBuffer();
        const wb = XLSX.read(buf, { type: "array" });
        const sheetNames = wb.SheetNames;
        setExcelSheets(sheetNames);

        if (sheetNames.length === 1) {
          // Один лист — сразу загружаем превью
          setSelectedSheet(sheetNames[0]);
          loadSheetPreview(next, sheetNames[0]);
        } else {
          // Много листов — ждём выбора пользователя
          setCsvPreview({ headers: [], rows: [], rowCount: 0, riskPct: null });
        }

      } else {
        // --- CSV (единый поток через DataEnrichment, как Excel) ---
        setSheetPreview(null);
        const text = await next.text();
        const lines = text.split(/\r?\n/).filter(Boolean);
        if (lines.length === 0) return;
        const headers = lines[0].split(",").map((h) => h.trim());
        const riskIdx = headers.indexOf("risk_flag");
        let riskPct = null;
        if (riskIdx >= 0) {
          const vals = lines.slice(1).map((l) => Number((l.split(",")[riskIdx] || "").trim()));
          const valid = vals.filter((v) => Number.isFinite(v));
          if (valid.length) { const risky = valid.filter((v) => v === 1).length; riskPct = (risky / valid.length) * 100; }
        }

        const allRows = lines.slice(1).map((l) => {
          const cells = l.split(",").map((c) => c.trim());
          const obj = {};
          headers.forEach((h, i) => {
            const v = cells[i] ?? "";
            obj[h] = v === "" ? "" : (isNaN(v) ? v : Number(v));
          });
          return obj;
        });

        // CSV идёт в rawExcelData → DataEnrichment → csvData (как Excel)
        const filtered = filterServiceCols(allRows);
        setRawExcelData(filtered);

        // Превью из первых 5 строк (без фильтрации служебных для совместимости)
        setCsvPreview({ headers, rows: lines.slice(1, 6).map((l) => l.split(",").map(c => c.trim())), rowCount: Math.max(lines.length - 1, 0), riskPct });
      }
    } catch (err) {
      console.error(err);
      setError("Ошибка чтения файла: " + err.message);
      setBusy(false);
    }
  }

  function renderCorrelationTable() {
    if (!corrResult?.correlation_matrix) return null;
    const matrix = corrResult.correlation_matrix;
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
          <thead><tr><th>feature</th>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {cols.map((r) => (
              <tr key={r}>
                <td><b>{r}</b></td>
                {cols.map((c) => {
                  const v = Number(matrix[r]?.[c] ?? 0);
                  return <td key={`${r}_${c}`} className={getCellClass(v)}>{v.toFixed(2)}</td>;
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
      <AnalysisSidebar onRun={onRunAnalysis} busy={busy} disabled={!targetSelected} />

      <div className="analysis-main">
        <div className="card">
          <h2>1) Загрузка данных</h2>
          <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileChange} />
          <p>Файл: {file?.name || "не выбран"}</p>

          {/* --- ВЫБОР ЛИСТА EXCEL --- */}
          {excelSheets.length > 1 && (
            <div className="sheet-selector" style={{ marginTop: 12, padding: 12, background: "var(--bg-secondary)", borderRadius: 8 }}>
              <label><b>Файл содержит {excelSheets.length} листов. Выберите:</b>
                <select value={selectedSheet} onChange={(e) => onSheetSelect(e.target.value)}
                  style={{ width: "100%", marginTop: 6, padding: 8, borderRadius: 6, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)" }}>
                  <option value="">— Выберите лист —</option>
                  {excelSheets.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </label>
            </div>
          )}
          {/* ------------------------------- */}

          {busy && <p>⏳ Загрузка и анализ файла...</p>}

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
              groupLabel={sheetPreview?.group_label}
              detectedGroup={sheetPreview?.detected_group}
              onConfirm={onEnrichmentConfirm}
              onSkip={onEnrichmentSkip}
            />
          )}
          {/* --------------------------- */}

          {csvPreview.rowCount > 0 && (
            <>
              <p>Строк: <b>{csvPreview.rowCount}</b>
                {csvPreview.riskPct != null && <> | Доля risk_flag=1: <b>{csvPreview.riskPct.toFixed(1)}%</b></>}
              </p>
              <div className="table-wrap">
                <table>
                  <thead><tr>{csvPreview.headers.map((h) => <th key={h}>{h}</th>)}</tr></thead>
                  <tbody>{csvPreview.rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={`${i}_${j}`}>{c}</td>)}</tr>)}</tbody>
                </table>
              </div>
            </>
          )}
        </div>

        {/* === ВЫБОР ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (как в Streamlit) === */}
        {csvData && csvData.length > 0 && !targetSelected && targetCandidates.length > 0 && !sheetPreview && !rawExcelData && (
          <div className="card">
            <h2>🎯 Выбор целевой переменной</h2>
            <p>Всего записей: <b>{csvData.length}</b> | Колонки: <b>{targetCandidates.length}</b></p>

            <label><b>Колонка для предсказания:</b>
              <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} style={{ width: "100%", padding: 8, marginTop: 4, borderRadius: 6, border: "1px solid var(--border)", background: "var(--bg)", color: "var(--text)" }}>
                <option value="">— Выберите колонку —</option>
                {targetCandidates.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>

            {targetStats && (
              <div className="muted" style={{ fontSize: 13, marginTop: 8 }}>
                <p style={{ margin: "4px 0" }}>Тип: <b>{targetStats.type}</b> | Уникальных: <b>{targetStats.unique}</b>
                  {targetStats.min != null && <> | Диапазон: {targetStats.min} — {targetStats.max}</>}
                </p>
                {targetStats.needsBinarize && (
                  <p className="muted" style={{ color: "var(--primary)" }}>
                    ℹ️ Колонка содержит более 2 значений — будет преобразована в бинарную (порог = медиана)
                  </p>
                )}
              </div>
            )}

            <div className="row" style={{ marginTop: 12 }}>
              <button className="primary" disabled={!targetColumn} onClick={() => setTargetSelected(true)}>
                ✅ Подтвердить выбор цели
              </button>
              <button onClick={() => {
                setCsvData(null); setFile(null); setTargetColumn("");
                setTargetSelected(false); setSheetPreview(null);
              }}>
                🔄 Сбросить данные
              </button>
            </div>
          </div>
        )}

        {error && <p className="error">{error}</p>}

        {/* Результаты ПОСЛЕ нажатия кнопки */}
        {analysisResult && <AnalysisResults result={analysisResult} />}

        {/* Быстрые инструменты (показываются только если анализ ещё не запущен) */}
        {!analysisResult && (
          <>
            <div className="card">
              <h2>2) Корреляционный анализ (быстрый)</h2>
              <button onClick={onCorrelation} disabled={!file || busy}>Запустить корреляцию</button>
              {corrResult ? (
                <>
                  <p>Размер: <b>{corrResult.n_rows}</b> строк × <b>{corrResult.n_columns}</b> колонок</p>
                  {renderCorrelationTable()}
                  {/* Plotly Heatmap */}
                  {corrResult.heatmap && (
                    <div className="plots-grid" style={{ marginTop: 12 }}>
                      <Plot data={corrResult.heatmap.data} layout={corrResult.heatmap.layout} config={{ responsive: true }} style={{ width: "100%" }} />
                    </div>
                  )}
                </>
              ) : <p className="muted">После запуска появится корреляционная матрица.</p>}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState("checking");

  useEffect(() => {
    healthcheck().then(() => setHealth("ok")).catch(() => setHealth("down"));
  }, []);

  return (
    <HashRouter>
      <ThemeToggle />
      <main className="container">
        <h1>АРМ исследователя — Мониторинг академических рисков</h1>
        <p className="muted">React MFE · FastAPI API · ml_core</p>
        <p>API: <span className={health === "ok" ? "ok" : health === "down" ? "error" : ""}>{health}</span></p>
        <Tabs />
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/imputation" element={<Imputation />} />
          <Route path="/crosstab" element={<Crosstab />} />
          <Route path="/timeseries" element={<TimeSeries />} />
          <Route path="/composite" element={<CompositeScore />} />
          <Route path="/drift" element={<DriftCheck />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/combinations" element={<FeatureCombinations />} />
          <Route path="/subset" element={<SubsetSelect />} />
        </Routes>
      </main>
    </HashRouter>
  );
}
