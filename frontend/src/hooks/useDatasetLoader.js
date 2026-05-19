import { useState, useEffect, useCallback, useRef } from "react";
import { useSharedData } from "./useDatasetStore";

export function useDatasetLoader() {
  const {
    loadData,
    hasShared,
    isLoading: storeLoading,
    datasetId,
  } = useSharedData();
  const [data, setData] = useState(null);
  const [columns, setColumns] = useState([]);
  const [loading, setLoading] = useState(true);
  const isMounted = useRef(true);
  const loadingRef = useRef(false);

  const refreshData = useCallback(async () => {
    if (loadingRef.current) return;

    if (!hasShared) {
      if (isMounted.current) {
        setData(null);
        setColumns([]);
        setLoading(false);
      }
      return;
    }

    loadingRef.current = true;
    if (isMounted.current) setLoading(true);

    try {
      const dataset = await loadData(true);
      if (isMounted.current && dataset) {
        setData(dataset);
        setColumns(dataset.length > 0 ? Object.keys(dataset[0]) : []);
      }
    } catch (err) {
      console.error("Failed to load data:", err);
    } finally {
      if (isMounted.current) setLoading(false);
      loadingRef.current = false;
    }
  }, [hasShared, loadData]);

  useEffect(() => {
    isMounted.current = true;
    refreshData();

    return () => {
      isMounted.current = false;
    };
  }, [datasetId]); // ← ТОЛЬКО при смене ID датасета, не при каждом рендере

  return {
    data,
    columns,
    loading: loading || storeLoading,
    hasData: !!data && data.length > 0,
    hasShared,
    updateData: useSharedData().updateData,
  };
}
