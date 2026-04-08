import { useState, useEffect } from "react";

/**
 * Компонент для настройки маппинга строковых колонок из Excel.
 * Позволяет выбрать Ordinal (свои числа) или One-Hot (бинарные колонки).
 */
export default function SheetMapper({ preview, onConfirm, onSkip }) {
    const { columns, detected_group: detectedType } = preview;

    // Находим строковые колонки (объекты)
    const stringCols = columns.filter(col => col.dtype === "object" || col.dtype === "string");

    // Состояние конфигурации: { colName: { type: "ordinal"|"one_hot"|"skip", map: {}, separator: "," } }
    const [config, setConfig] = useState({});

    // Инициализация дефолтных значений при загрузке
    useEffect(() => {
        const initialConfig = {};
        stringCols.forEach(col => {
            // Эвристика: если много значений и есть разделители — скорее всего Multiple Choice
            const isMulti = detectedType === "multiple_choice" || col.unique_values.some(v => String(v).includes(";"));

            if (isMulti) {
                initialConfig[col.name] = { type: "split", separator: ";", unique: col.unique_values };
            } else if (col.unique_values.length <= 10) {
                // Для малого кол-ва значений предлагаем Ordinal по умолчанию
                initialConfig[col.name] = {
                    type: "ordinal",
                    map: {}, // Пусто пока пользователь не заполнит
                    unique: col.unique_values
                };
            } else {
                // Для большого кол-ва значений (например, имена) лучше One-Hot или Skip
                initialConfig[col.name] = { type: "one_hot", unique: col.unique_values };
            }
        });
        setConfig(initialConfig);
    }, [preview]);

    // Обработчик изменения типа кодирования
    const handleTypeChange = (colName, newType) => {
        setConfig(prev => ({
            ...prev,
            [colName]: { ...prev[colName], type: newType }
        }));
    };

    // Обработчик изменения значения в Ordinal Map
    const handleOrdinalChange = (colName, valueKey, numberVal) => {
        setConfig(prev => {
            const currentCol = prev[colName];
            return {
                ...prev,
                [colName]: {
                    ...currentCol,
                    map: { ...currentCol.map, [valueKey]: numberVal }
                }
            };
        });
    };

    // Обработчик изменения разделителя
    const handleSeparatorChange = (colName, sep) => {
        setConfig(prev => ({
            ...prev,
            [colName]: { ...prev[colName], separator: sep }
        }));
    };

    // Формирование финального конфига для отправки (убираем служебное поле unique)
    const handleApply = () => {
        const finalConfig = {};
        Object.entries(config).forEach(([col, settings]) => {
            if (settings.type === "skip") return;
            finalConfig[col] = {
                type: settings.type,
                map: settings.type === "ordinal" ? settings.map : undefined,
                separator: settings.type === "split" ? settings.separator : undefined
            };
        });
        onConfirm(finalConfig);
    };

    if (stringCols.length === 0) {
        return <p className="muted">Строковых колонок не найдено. Переходим к анализу...</p>;
    }

    return (
        <div className="card sheet-mapper">
            <h2>⚙️ Настройка обработки текстовых данных</h2>
            <p className="muted">
                Категория листа: <b>{preview.group_label || detectedType}</b>
                {" · "}Найдено колонок с текстом: <b>{stringCols.length}</b>.
                Укажите, как интерпретировать их значения (числовая ценность или разделение).
            </p>

            {stringCols.map(col => {
                const settings = config[col.name] || {};
                return (
                    <div key={col.name} className="mapper-col">
                        <div className="mapper-header">
                            <label>
                                <b>{col.name}</b> ({col.unique_values.length} уникальных значений)
                            </label>
                            <select
                                value={settings.type || "one_hot"}
                                onChange={e => handleTypeChange(col.name, e.target.value)}
                            >
                                <option value="ordinal">🔢 Порядковая (Ordinal)</option>
                                <option value="one_hot">📊 Бинарная (One-Hot)</option>
                                <option value="split">✂️ Разделить (Multiple Choice)</option>
                                <option value="skip">🗑️ Пропустить</option>
                            </select>
                        </div>

                        {/* UI для Ordinal: ввод чисел для каждого значения */}
                        {settings.type === "ordinal" && (
                            <div className="ordinal-grid">
                                {col.unique_values.map(val => (
                                    <div key={val} className="ordinal-item">
                                        <span>{String(val)}</span>
                                        <span>=</span>
                                        <input
                                            type="number"
                                            value={settings.map?.[val] ?? ""}
                                            placeholder="0"
                                            onChange={e => handleOrdinalChange(col.name, val, e.target.value)}
                                        />
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* UI для Split: выбор разделителя */}
                        {settings.type === "split" && (
                            <div className="split-controls">
                                <label>Разделитель: </label>
                                <select
                                    value={settings.separator || ","}
                                    onChange={e => handleSeparatorChange(col.name, e.target.value)}
                                >
                                    <option value=";">Точка с запятой (;)</option>
                                    <option value=",">Запятая (,)</option>
                                </select>
                                <p className="muted">Будет создано {col.unique_values.length} новых колонок (0/1).</p>
                            </div>
                        )}
                    </div>
                );
            })}

            <div className="row" style={{ marginTop: 20 }}>
                <button className="primary" onClick={handleApply}>✅ Применить и загрузить данные</button>
                <button onClick={onSkip}>⏭️ Пропустить (авто-обработка)</button>
            </div>

            <style>{`
                .sheet-mapper { border-left: 4px solid var(--primary); }
                .mapper-col {
                    background: var(--bg-secondary);
                    padding: 12px;
                    margin-bottom: 10px;
                    border-radius: 8px;
                }
                .mapper-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
                .ordinal-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
                .ordinal-item { display: flex; align-items: center; gap: 8px; font-size: 14px; }
                .ordinal-item input { width: 60px; padding: 4px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; }
                .split-controls { display: flex; align-items: center; gap: 10px; font-size: 14px; }
            `}</style>
        </div>
    );
}
