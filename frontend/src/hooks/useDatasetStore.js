import { create } from "zustand";
import { persist } from "zustand/middleware";
import Dexie from "dexie";
import { useEffect } from "react";

// Инициализация IndexedDB
const db = new Dexie("ARM_Datasets");
db.version(1).stores({
  datasets: "id, timestamp, rowCount",
  experiments: "++id, name, timestamp",
});

export const useDatasetStore = create(
  persist(
    (set, get) => ({
      // Текущее активное состояние (in-memory)
      currentData: [],
      currentColumns: [],
      currentDatasetId: null,
      metadata: {},
      isLoading: false,
      // Основной метод обновления
      setData: (data, metadata = {}) => {
        if (!data || !Array.isArray(data)) {
          set({ currentData: [], currentColumns: [], currentDatasetId: null });
          return;
        }

        const columns = data.length > 0 ? Object.keys(data[0]) : [];
        const datasetId = metadata.id || `ds_${Date.now()}`;

        const newState = {
          currentData: data,
          currentColumns: columns,
          currentDatasetId: datasetId,
          metadata: {
            ...metadata,
            rowCount: data.length,
            lastUpdated: new Date().toISOString(),
          },
        };

        set(newState);

        // Сохраняем в IndexedDB (асинхронно)
        db.datasets
          .put({
            id: datasetId,
            data: data,
            columns: columns,
            metadata: newState.metadata,
            timestamp: Date.now(),
          })
          .catch((err) => console.error("Failed to save to IndexedDB:", err));
      },
      //Восстановление данных при монтировании
      hydrate: async () => {
        const { currentDatasetId } = get();
        if (currentDatasetId) {
          set({ isLoading: true });
          await get().loadFromDB(currentDatasetId);
          set({ isLoading: false });
        }
      },

      loadFromDB: async (datasetId) => {
        try {
          set({ isLoading: true });
          const record = await db.datasets.get(datasetId);
          if (record) {
            set({
              currentData: record.data,
              currentColumns: record.columns,
              currentDatasetId: record.id,
              metadata: record.metadata,
            });
            return true;
          }
        } catch (err) {
          console.error("Failed to load from IndexedDB:", err);
        } finally {
          set({ isLoading: false });
        }
        return false;
      },

      clearData: () => {
        set({
          currentData: [],
          currentColumns: [],
          currentDatasetId: null,
          metadata: {},
        });
      },
    }),

    {
      name: "arm-current-dataset", // для persist в localStorage (только метаданные)
      partialize: (state) => ({
        currentDatasetId: state.currentDatasetId,
        metadata: state.metadata,
      }),
    },
  ),
);

// Экспортируем хук
export const useSharedData = () => {
  const store = useDatasetStore();
  useEffect(() => {
    if (!store.hydrated) {
      store.hydrate();
      // можно добавить флаг hydrated в store
    }
  }, []);
  return {
    data: store.currentData,
    columns: store.currentColumns,
    hasShared: store.currentData.length > 0,
    datasetId: store.currentDatasetId,
    metadata: store.metadata,
    isLoading: store.isLoading,
    updateData: store.setData,
    loadFromDB: store.loadFromDB,
    clearData: store.clearData,
    hydrate: store.hydrate,
  };
};
