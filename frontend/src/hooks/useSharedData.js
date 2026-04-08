import { useState, useCallback } from "react";

const SHARED_DATA_KEY = "shared_dataset";

/**
 * Хук для доступа к общему датасету.
 * Использует sessionStorage — данные живут до закрытия вкладки.
 * СИНХРОННОЕ чтение при инициализации — данные доступны сразу.
 */
export function useSharedData() {
  const [state, setState] = useState(() => {
    try {
      const raw = sessionStorage.getItem(SHARED_DATA_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && parsed.length > 0) {
          return { data: parsed, columns: Object.keys(parsed[0]), hasShared: true };
        }
      }
    } catch { /* ignore */ }
    return { data: null, columns: [], hasShared: false };
  });

  const updateData = useCallback((newData) => {
    if (!newData || !newData.length) return;
    setState({ data: newData, columns: Object.keys(newData[0]), hasShared: true });
    sessionStorage.setItem(SHARED_DATA_KEY, JSON.stringify(newData));
  }, []);

  const clearData = useCallback(() => {
    setState({ data: null, columns: [], hasShared: false });
    sessionStorage.removeItem(SHARED_DATA_KEY);
  }, []);

  return {
    data: state.data,
    columns: state.columns,
    hasShared: state.hasShared,
    updateData,
    clearData,
  };
}
