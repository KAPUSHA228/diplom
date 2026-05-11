// hooks/useDatasetLoader.js
import { useState, useEffect } from "react";
import { useSharedData } from "./useDatasetStore";

export function useDatasetLoader() {
  const { loadData, hasShared, isLoading: storeLoading } = useSharedData();
  const [data, setData] = useState(null);
  const [columns, setColumns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      if (loaded) return;
      setLoading(true);
      const dataset = await loadData();
      if (dataset) {
        setData(dataset);
        setColumns(dataset.length > 0 ? Object.keys(dataset[0]) : []);
        setLoaded(true);
      }
      setLoading(false);
    };

    if (hasShared) {
      fetchData();
    } else {
      setLoading(false);
    }
  }, [hasShared, loadData, loaded]);

  return {
    data,
    columns,
    loading: loading || storeLoading,
    hasData: !!data && data.length > 0,
    hasShared,
    updateData: useSharedData().updateData,
  };
}
