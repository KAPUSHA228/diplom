import { useState, useEffect } from "react";
import { useDatasetStore } from "../hooks/useDatasetStore";

export default function DatasetHistory({ onLoad, refreshTrigger }) {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(false);

  const getAllDatasets = useDatasetStore((state) => state.getAllDatasets);
  const loadDatasetById = useDatasetStore((state) => state.loadDatasetById);
  const deleteDatasetById = useDatasetStore((state) => state.deleteDatasetById);
  const currentDatasetId = useDatasetStore((state) => state.currentDatasetId);

  const refreshList = async () => {
    setLoading(true);
    const list = await getAllDatasets();
    setDatasets(list);
    setLoading(false);
  };

  useEffect(() => {
    refreshList();
  }, [refreshTrigger]);

  const handleLoad = async (id) => {
    const data = await loadDatasetById(id);
    if (data && onLoad) {
      onLoad(data);
    }
  };

  const handleDelete = async (id, e) => {
    e.stopPropagation();
    if (window.confirm("Удалить этот датасет?")) {
      await deleteDatasetById(id);
      refreshList();
    }
  };

  if (datasets.length === 0 && !loading) {
    return (
      <div className="card" style={{ marginTop: 12 }}>
        <h3>История загрузок</h3>
        <p className="muted">Нет сохранённых датасетов. Загрузите файл.</p>
      </div>
    );
  }

  return (
    <div className="card" style={{ marginTop: 12 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h3 style={{ margin: 0 }}>
          История загрузок (последние {datasets.length})
        </h3>
        <button
          onClick={refreshList}
          disabled={loading}
          style={{ fontSize: 12 }}
        >
          Обновить
        </button>
      </div>

      <div style={{ marginTop: 8 }}>
        {datasets.map((ds) => (
          <div
            key={ds.id}
            onClick={() => handleLoad(ds.id)}
            style={{
              padding: "8px 12px",
              marginBottom: 6,
              borderRadius: 6,
              background:
                currentDatasetId === ds.id
                  ? "var(--primary)"
                  : "var(--bg-secondary)",
              color: currentDatasetId === ds.id ? "white" : "var(--text)",
              cursor: "pointer",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              transition: "all 0.2s",
            }}
          >
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 500 }}>
                {ds.metadata.name ||
                  `Датасет от ${new Date(ds.timestamp).toLocaleString()}`}
              </div>
              <div style={{ fontSize: 11, opacity: 0.7 }}>
                {ds.metadata.rowCount || ds.data?.length || 0} строк ·{" "}
                {ds.columns?.length || 0} колонок
              </div>
            </div>
            <button
              onClick={(e) => handleDelete(ds.id, e)}
              style={{
                background: "transparent",
                border: "none",
                color: currentDatasetId === ds.id ? "white" : "#e74c3c",
                cursor: "pointer",
                fontSize: 16,
              }}
              title="Удалить"
            >
              🗑️
            </button>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 12, fontSize: 12, color: "var(--text-muted)" }}>
        Кликните по датасету — он загрузится в систему
      </div>
    </div>
  );
}
