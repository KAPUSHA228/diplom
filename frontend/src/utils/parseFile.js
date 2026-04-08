import * as XLSX from "xlsx";

/**
 * Парсит CSV или Excel файл в массив объектов (JSON).
 * Возвращает Promise<{ headers, rows, allData }>.
 */
export async function parseFile(file) {
  if (!file) return null;

  const isExcel = file.name.endsWith(".xlsx") || file.name.endsWith(".xls");

  if (isExcel) {
    const buf = await file.arrayBuffer();
    const wb = XLSX.read(buf, { type: "array" });
    // Берём первый лист
    const ws = wb.Sheets[wb.SheetNames[0]];
    const allData = XLSX.utils.sheet_to_json(ws, { defval: "" });

    if (!allData.length) return { headers: [], rows: [], allData: [] };

    const headers = Object.keys(allData[0]);
    const previewRows = allData.slice(0, 5).map((r) => headers.map((h) => String(r[h] ?? "")));

    return { headers, rows: previewRows, allData };
  }

  // CSV
  const text = await file.text();
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (!lines.length) return { headers: [], rows: [], allData: [] };

  const headers = lines[0].split(",").map((h) => h.trim());
  const previewRows = lines.slice(1, 6).map((l) => l.split(","));

  const allData = lines.slice(1).map((l) => {
    const cells = l.split(",").map((c) => c.trim());
    const obj = {};
    headers.forEach((h, i) => {
      const v = cells[i] ?? "";
      obj[h] = v === "" ? "" : isNaN(v) ? v : Number(v);
    });
    return obj;
  });

  return { headers, rows: previewRows, allData };
}
