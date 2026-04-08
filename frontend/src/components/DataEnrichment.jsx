import { useState } from "react";

const STRATEGIES = [
    { value: "auto", label: "🤖 Авто (умный выбор)" },
    { value: "median", label: "📊 Медиана" },
    { value: "mean", label: "🔢 Среднее" },
    { value: "interpolate", label: "↕️ Интерполяция" },
    { value: "drop_rows", label: "🗑️ Удалить строки" },
];

export default function DataEnrichment({ groupLabel, detectedGroup, onConfirm, onSkip }) {
    const [strategy, setStrategy] = useState("auto");
    const [threshold, setThreshold] = useState(30.0);

    return (
        <div className="card data-enrichment">
            <h2>🧪 Обогащение данных</h2>
            <p className="muted">
                Категория листа: <b>{groupLabel || detectedGroup}</b>
            </p>

            <div className="enrichment-controls">
                <label>
                    <b>Стратегия обработки пропусков:</b>
                    <div className="strategy-grid">
                        {STRATEGIES.map(s => (
                            <label key={s.value} className={`strategy-option ${strategy === s.value ? "selected" : ""}`}>
                                <input
                                    type="radio"
                                    name="imputation-strategy"
                                    value={s.value}
                                    checked={strategy === s.value}
                                    onChange={() => setStrategy(s.value)}
                                />
                                {s.label}
                            </label>
                        ))}
                    </div>
                </label>

                <label>
                    <b>Порог выбросов (IQR):</b> {threshold}
                    <input
                        type="range"
                        min="10"
                        max="50"
                        step="1"
                        value={threshold}
                        onChange={e => setThreshold(Number(e.target.value))}
                    />
                </label>
            </div>

            <div className="row" style={{ marginTop: 16 }}>
                <button className="primary" onClick={() => onConfirm({ strategy, threshold })}>
                    ✅ Применить обогащение
                </button>
                <button onClick={onSkip}>⏭️ Пропустить</button>
            </div>

            <style>{`
                .data-enrichment { border-left: 4px solid var(--primary); }
                .enrichment-controls { display: flex; flex-direction: column; gap: 16px; margin-top: 12px; }
                .strategy-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 8px; margin-top: 8px; }
                .strategy-option {
                    display: flex; align-items: center; gap: 8px;
                    padding: 8px 12px; border-radius: 6px;
                    border: 1px solid var(--border); cursor: pointer;
                    transition: background 0.2s, border-color 0.2s;
                }
                .strategy-option:hover { background: var(--bg-secondary); }
                .strategy-option.selected {
                    border-color: var(--primary);
                    background: color-mix(in srgb, var(--primary) 15%, transparent);
                }
                .strategy-option input { margin: 0; }
            `}</style>
        </div>
    );
}
