import { useState } from "react";
import Plot from "react-plotly.js";
import { buildCrosstab } from "../api";
import { parseFile } from "../utils/parseFile";
import { useSharedData } from "../hooks/useSharedData";

export default function Crosstab() {
  const { data: sharedData, columns: sharedCols, hasShared } = useSharedData();
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [rowVar, setRowVar] = useState("");
  const [colVar, setColVar] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const activeData = fileData || sharedData;
  const activeCols = fileData ? Object.keys(fileData[0] || {}) : sharedCols;

  function initData(data) {
    if (!data?.length) return;
    const cols = Object.keys(data[0]);
    setRowVar(cols[0] || "");
    setColVar(cols[1] || "");
  }

  async function onFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f); setResult(null);
    try {
      const parsed = await parseFile(f);
      setFileData(parsed?.allData || null);
      initData(parsed?.allData);
    } catch { /* ignore */ }
  }

  // Инициализация колонок при загрузке shared data
  useState(() => {
    if (hasShared && !fileData && !rowVar) initData(sharedData);
  });

  async function onRun() {
    if (!activeData || !rowVar || !colVar) return;
    setBusy(true); setError("");
    try {
      const res = await buildCrosstab(activeData, rowVar, colVar);
      setResult(res);
    } catch (err) { setError("Ошибка: " + err.message); }
    finally { setBusy(false); }
  }

  return (
    <div className="card">
      <h2>📈 Кросс-таблица (χ²-тест)</h2>

      <div style={{ marginBottom: 12 }}>
        {hasShared && !fileData && (
          <div className="ok" style={{ padding: 8, borderRadius: 6, background: "var(--bg-secondary)", marginBottom: 8 }}>
            ✅ Используются данные с главной: <b>{sharedData.length} строк</b>, {activeCols.length} колонок
          </div>
        )}
        <input type="file" accept=".csv,.xlsx,.xls" onChange={onFileChange} />
      </div>

      {activeCols.length > 0 && (
        <div className="row" style={{ marginTop: 8 }}>
          <label>Строки: </label>
          <select value={rowVar} onChange={(e) => setRowVar(e.target.value)}>
            {activeCols.map((h) => <option key={h} value={h}>{h}</option>)}
          </select>
          <label>Столбцы: </label>
          <select value={colVar} onChange={(e) => setColVar(e.target.value)}>
            {activeCols.map((h) => <option key={h} value={h}>{h}</option>)}
          </select>
          <button onClick={onRun} disabled={busy}>Построить</button>
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {result?.table && (
        <>
          <div className="table-wrap" style={{ marginTop: 12 }}>
            <h3>Таблица сопряжённости</h3>
            <table>
              <thead><tr><th></th>{Object.keys(result.table).map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {Object.entries(result.table).map(([row, vals]) => (
                  <tr key={row}><td><b>{row}</b></td>{Object.values(vals).map((v, i) => <td key={i}>{typeof v === "number" ? v.toFixed(2) : v}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Heatmap */}
          {(() => {
            const rows = Object.keys(result.table);
            const cols = Object.keys(result.table[rows[0]] || {});
            const z = rows.map(r => cols.map(c => result.table[r][c] ?? 0));
            return (
              <div style={{ marginTop: 16 }}>
                <h3>Heatmap</h3>
                <Plot data={[{ z, x: cols, y: rows, type: "heatmap", colorscale: "YlOrRd", text: z.map(row => row.map(v => v.toFixed(1))), texttemplate: "%{text}", hovertemplate: "%{y} × %{x}: %{z:.1f}<extra></extra>" }]} layout={{ margin: { l: 100, r: 20, t: 30, b: 60 }, height: 300 + rows.length * 30 }} config={{ responsive: true, displayModeBar: false }} style={{ width: "100%" }} />
              </div>
            );
          })()}

          {/* Stacked Bar */}
          {(() => {
            const rows = Object.keys(result.table);
            const cols = Object.keys(result.table[rows[0]] || {});
            return (
              <div style={{ marginTop: 16 }}>
                <h3>Stacked Bar</h3>
                <Plot data={cols.map((col) => ({ name: col, type: "bar", x: rows, y: rows.map(r => result.table[r]?.[col] ?? 0) }))} layout={{ barmode: "stack", margin: { l: 50, r: 20, t: 30, b: 60 }, height: 350, showlegend: true, legend: { orientation: "h", y: -0.3 } }} config={{ responsive: true, displayModeBar: false }} style={{ width: "100%" }} />
              </div>
            );
          })()}
        </>
      )}

      {result?.chi2_test && (
        <p style={{ marginTop: 12 }}>χ² тест: p-value = <b>{result.chi2_test.p_value?.toFixed(4)}</b> {result.chi2_test.significant ? "✅ Значимо" : "❌ Не значимо"}</p>
      )}
    </div>
  );
}
