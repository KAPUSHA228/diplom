import { useState, useEffect, useMemo } from "react";

const LIKERT_ORDER = [
  "полностью не согласен",
  "совершенно не согласен",
  "не согласен",
  "скорее не согласен",
  "затрудняюсь ответить",
  "нейтрально",
  "скорее согласен",
  "согласен",
  "полностью согласен",
  "совершенно согласен",
];

function autoOrdinalMap(uniqueValues = []) {
  const asStrings = uniqueValues.map((v) => String(v).trim());
  if (!asStrings.length) return {};

  const asNumbers = asStrings.map((v) => Number(v));
  const numericLike = asNumbers.every((n) => Number.isFinite(n));
  if (numericLike) {
    return Object.fromEntries(asStrings.map((v) => [v, Number(v)]));
  }

  // Если ответы похожи на шкалу Лайкерта, задаём предсказуемый порядок.
  const scored = asStrings.map((value) => {
    const low = value.toLowerCase();
    const idx = LIKERT_ORDER.findIndex((token) => low.includes(token));
    return { value, score: idx >= 0 ? idx + 1 : null };
  });
  if (scored.every((x) => x.score != null)) {
    return Object.fromEntries(scored.map((x) => [x.value, x.score]));
  }

  // Фолбэк: стабильная нумерация по алфавиту.
  const sorted = [...new Set(asStrings)].sort((a, b) =>
    a.localeCompare(b, "ru"),
  );
  return Object.fromEntries(sorted.map((value, i) => [value, i + 1]));
}

/**
 * Компонент для настройки маппинга строковых колонок из Excel.
 * Позволяет выбрать Ordinal (свои числа) или One-Hot (бинарные колонки).
 */
export default function SheetMapper({ preview, onConfirm, onSkip }) {
  const { columns, detected_group: detectedType, sheet_name } = preview;
  const isMultipleSheet = detectedType === "multiple_choice";

  // Находим строковые колонки (объекты)
  const stringCols = useMemo(
    () =>
      columns.filter((col) => col.dtype === "object" || col.dtype === "string"),
    [columns],
  );
  const uniqueValuesByCol = useMemo(
    () =>
      Object.fromEntries(
        stringCols.map((col) => [col.name, col.unique_values || []]),
      ),
    [stringCols],
  );

  // Состояние конфигурации: { colName: { type: "ordinal"|"one_hot"|"skip", map: {}, separator: "," } }
  const [config, setConfig] = useState({});

  // Инициализация дефолтных значений при загрузке
  useEffect(() => {
    const initial = {};
    stringCols.forEach((col) => {
      // Эвристика: если много значений и есть разделители — скорее всего Multiple Choice
      if (isMultipleSheet) {
        initial[col.name] = {
          type: "split",
          separator: ";",
          encoding: "onehot",
          imputation: "mode",
        };
      } else {
        initial[col.name] = {
          type: "ordinal",
          encoding: "ordinal",
          imputation: "mode",
        };
      }
    });

    setConfig(initial);
  }, [stringCols, isMultipleSheet]);

  // Обработчик изменения типа кодирования
  const handleTypeChange = (colName, newType) => {
    setConfig((prev) => ({
      ...prev,
      [colName]: { ...prev[colName], type: newType },
    }));
  };

  //   useEffect(() => {
  //     setConfig((prev) => {
  //       const next = { ...prev };
  //       for (const [colName, settings] of Object.entries(next)) {
  //         const currentType = settings?.type;
  //         if (!allowedTypes.includes(currentType)) {
  //           const fallbackType = isMultipleSheet ? "split" : "ordinal";
  //           next[colName] = {
  //             ...settings,
  //             type: fallbackType,
  //             encoding: fallbackType === "ordinal" ? "ordinal" : "onehot",
  //           };
  //         }
  //       }
  //       return next;
  //     });
  //   }, [allowedTypes, isMultipleSheet]);

  //   // Обработчик изменения значения в Ordinal Map
  const handleOrdinalChange = (colName, valueKey, numberVal) => {
    setConfig((prev) => {
      const currentCol = prev[colName];
      return {
        ...prev,
        [colName]: {
          ...currentCol,
          map: { ...currentCol.map, [valueKey]: numberVal },
        },
      };
    });
  };

  // Обработчик изменения разделителя
  const handleSeparatorChange = (colName, sep) => {
    setConfig((prev) => ({
      ...prev,
      [colName]: { ...prev[colName], separator: sep },
    }));
  };

  // Формирование финального конфига для отправки (убираем служебное поле unique)
  const handleApply = () => {
    const columnsConfig = {};

    Object.entries(config).forEach(([colName, settings]) => {
      if (settings.type === "skip") {
        columnsConfig[colName] = { type: "skip" };
        return;
      }

      columnsConfig[colName] = {
        type: settings.type,
        separator: settings.separator || ";",
        encoding: settings.encoding || "onehot",
        imputation: settings.imputation || "mode",
        ...(settings.type === "ordinal" &&
          !isMultipleSheet && {
            map:
              settings.map || autoOrdinalMap(uniqueValuesByCol[colName] || []),
          }),
      };
    });

    const finalMappingConfig = {
      sheet_name: sheet_name,
      sheet_type: detectedType,
      columns: columnsConfig,
      global_settings: {
        drop_service_cols: true,
        handle_outliers: "iqr",
      },
      detected_group: detectedType,
    };

    console.log("Отправляем MappingConfig:", finalMappingConfig);
    onConfirm(finalMappingConfig);
  };

  if (stringCols.length === 0) {
    return (
      <p className="muted">
        Строковых колонок не найдено. Переходим к анализу...
      </p>
    );
  }

  return (
    <div className="card sheet-mapper">
      <h2>⚙️ Настройка обработки текстовых данных</h2>
      <p className="muted">
        Категория листа: <b>{preview.group_label || detectedType}</b>
        {" · "}Найдено колонок с текстом: <b>{stringCols.length}</b>. Укажите,
        как интерпретировать их значения (числовая ценность или разделение).
      </p>

      {stringCols.map((col) => {
        const settings = config[col.name] || {};
        return (
          <div key={col.name} className="mapper-col">
            <div className="mapper-header">
              <label>
                <b>{col.name}</b> ({col.unique_values.length} уникальных
                значений)
              </label>
              <select
                value={settings.type}
                onChange={(e) => handleTypeChange(col.name, e.target.value)}
              >
                {isMultipleSheet ? (
                  <option value="split">✂️ Разделить (Multiple Choice)</option>
                ) : (
                  <>
                    <option value="ordinal">🔢 Ordinal</option>
                    <option value="one_hot">📊 One-Hot</option>
                  </>
                )}
                <option value="skip">🗑️ Пропустить</option>
              </select>
            </div>

            {/* UI для Split: выбор разделителя */}
            {settings.type === "split" && (
              <div className="split-controls">
                <label>Разделитель: </label>
                <select
                  value={settings.separator}
                  onChange={(e) =>
                    handleSeparatorChange(col.name, e.target.value)
                  }
                >
                  <option value=";">Точка с запятой (;)</option>
                  <option value=",">Запятая (,)</option>
                </select>
                <p className="muted">
                  Будет создано несколько бинарных столбцов (0/1)
                </p>
              </div>
            )}

            {/* UI для Ordinal: ввод чисел для каждого значения */}
            {settings.type === "ordinal" && !isMultipleSheet && (
              <div className="ordinal-grid">
                <p
                  className="muted"
                  style={{ gridColumn: "1 / -1", margin: "0 0 6px 0" }}
                >
                  Оставьте поля пустыми для авто-шкалы (без раздувания таблицы).
                </p>
                {col.unique_values.map((val) => (
                  <div key={val} className="ordinal-item">
                    <span>{String(val)}</span>
                    <span>=</span>
                    <input
                      type="number"
                      value={settings.map?.[val] ?? ""}
                      placeholder="0"
                      onChange={(e) =>
                        handleOrdinalChange(col.name, val, e.target.value)
                      }
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}

      <div className="row" style={{ marginTop: 20 }}>
        <button className="primary" onClick={handleApply}>
          ✅ Применить и загрузить данные
        </button>
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
