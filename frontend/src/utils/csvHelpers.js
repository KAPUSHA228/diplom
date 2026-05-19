/**
 * Вспомогательные функции для работы с CSV/данными
 */

/**
 * Фильтрует служебные колонки из данных
 * @param {Array} data - массив объектов (строки данных)
 * @param {Set} excludeCols - Set с именами колонок для исключения
 * @returns {Array} отфильтрованные данные
 */
export function filterServiceCols(data, excludeCols) {
  if (!data || !data.length) return data;

  return data.map((row) => {
    const filtered = {};
    for (const [key, val] of Object.entries(row)) {
      if (!excludeCols.has(key.toLowerCase())) {
        filtered[key] = val;
      }
    }
    return filtered;
  });
}

/**
 * Скачивает массив объектов в CSV файл
 * @param {Array} data - массив объектов
 * @param {string} filename - имя файла (без расширения)
 */
export function downloadJSONAsCSV(data, filename) {
  if (!data || !data.length) return;

  const keys = Object.keys(data[0]);
  const csvContent =
    "data:text/csv;charset=utf-8," +
    [
      keys.join(","),
      ...data.map((row) =>
        keys.map((k) => JSON.stringify(row[k] || "")).join(","),
      ),
    ].join("\n");

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", filename + ".csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Преобразует объект профилей кластеров в массив для CSV
 * @param {Object} profiles - объект {cluster: {data}}
 * @returns {Array} массив строк для CSV
 */
export function clusterProfilesToArray(profiles) {
  if (!profiles) return [];
  return Object.entries(profiles).map(([cluster, data]) => ({
    cluster,
    ...data,
  }));
}
