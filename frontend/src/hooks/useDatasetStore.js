// useDatasetStore.js
import { create } from "zustand";
import { persist } from "zustand/middleware";
import Dexie from "dexie";
import { useEffect, useRef, useState, useCallback } from "react";

const db = new Dexie("ARM_Datasets");
db.version(1).stores({
  datasets: "id, timestamp, rowCount",
  experiments: "++id, name, timestamp",
});

export const useDatasetStore = create(
  persist(
    (set, get) => ({
      currentDatasetId: null,
      metadata: {},
      isLoading: false,
      hydrated: false,
      isHydrating: false,

      setData: async (data, metadata = {}) => {
        if (!data || !Array.isArray(data)) {
          set({ currentDatasetId: null, metadata: {} });
          return;
        }

        const columns = data.length > 0 ? Object.keys(data[0]) : [];
        const datasetId = metadata.id || `ds_${Date.now()}`;

        const record = {
          id: datasetId,
          data: data,
          columns: columns,
          metadata: {
            ...metadata,
            rowCount: data.length,
            lastUpdated: new Date().toISOString(),
          },
          timestamp: Date.now(),
        };

        await db.datasets.put(record);

        set({
          currentDatasetId: datasetId,
          metadata: record.metadata,
        });
      },

      getCurrentData: async () => {
        const { currentDatasetId } = get();
        if (!currentDatasetId) return null;

        try {
          const record = await db.datasets.get(currentDatasetId);
          return record?.data || null;
        } catch (err) {
          console.error("Failed to get data from IndexedDB:", err);
          return null;
        }
      },

      loadDataToMemory: async () => {
        const { currentDatasetId, isLoading } = get();
        if (!currentDatasetId || isLoading) return null;

        set({ isLoading: true });
        try {
          const record = await db.datasets.get(currentDatasetId);
          if (record?.data) {
            return record.data;
          }
        } catch (err) {
          console.error("Failed to load data:", err);
        } finally {
          set({ isLoading: false });
        }
        return null;
      },

      hydrate: async () => {
        const { hydrated, currentDatasetId, isHydrating } = get();
        if (hydrated || isHydrating) return;

        set({ isHydrating: true });

        if (currentDatasetId) {
          set({ isLoading: true });
          try {
            const record = await db.datasets.get(currentDatasetId);
            if (record) {
              set({
                metadata: record.metadata,
                hydrated: true,
                isLoading: false,
                isHydrating: false,
              });
            } else {
              set({ hydrated: true, isLoading: false, isHydrating: false });
            }
          } catch (err) {
            console.error("Hydration failed:", err);
            set({ hydrated: true, isLoading: false, isHydrating: false });
          }
        } else {
          set({ hydrated: true, isHydrating: false });
        }
      },

      clearData: () => {
        set({
          currentDatasetId: null,
          metadata: {},
          hydrated: false,
          isLoading: false,
          isHydrating: false,
        });
      },
    }),
    {
      name: "arm-current-dataset",
      partialize: (state) => ({
        currentDatasetId: state.currentDatasetId,
        metadata: state.metadata,
      }),
    },
  ),
);

export const useSharedData = () => {
  const store = useDatasetStore();
  const dataCache = useRef(null);
  const [isLoadingData, setIsLoadingData] = useState(false);
  const isHydratedRef = useRef(false);

  useEffect(() => {
    if (!isHydratedRef.current && !store.hydrated) {
      isHydratedRef.current = true;
      store.hydrate();
    }
  }, [store]);

  const loadData = useCallback(
    async (force = false) => {
      if (!store.currentDatasetId) return null;

      if (force || (!dataCache.current && !isLoadingData)) {
        setIsLoadingData(true);
        try {
          const loadedData = await store.getCurrentData();
          if (loadedData) {
            dataCache.current = loadedData;
          }
          return loadedData;
        } catch (err) {
          console.error("Failed to load data for display:", err);
          return null;
        } finally {
          setIsLoadingData(false);
        }
      }
      return dataCache.current;
    },
    [store, isLoadingData],
  );

  const getPreview = useCallback(
    async (limit = 100) => {
      if (!store.currentDatasetId) return [];
      try {
        const record = await db.datasets.get(store.currentDatasetId);
        return record?.data?.slice(0, limit) || [];
      } catch (err) {
        console.error("Failed to get preview:", err);
        return [];
      }
    },
    [store.currentDatasetId],
  );

  const getColumns = useCallback(() => {
    return store.metadata.columns || [];
  }, [store.metadata.columns]);

  useEffect(() => {
    dataCache.current = null;
  }, [store.currentDatasetId]);

  return {
    hasShared: !!store.currentDatasetId,
    datasetId: store.currentDatasetId,
    metadata: store.metadata,
    isLoading: store.isLoading || isLoadingData,

    updateData: store.setData,
    loadData,
    getPreview,
    getColumns,
    clearData: store.clearData,
  };
};
