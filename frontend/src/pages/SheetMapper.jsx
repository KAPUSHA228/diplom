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
  const [groups, setGroups] = useState([]);
  const [ungroupedCols, setUngroupedCols] = useState(new Set());

  // Группируем колонки по value_signature
  useEffect(() => {
    const groupsMap = {};

    stringCols.forEach((col) => {
      // Если колонка в ручном режиме — пропускаем
      if (ungroupedCols.has(col.name)) return;

      const sig = col.value_signature || col.name;
      if (!groupsMap[sig]) {
        groupsMap[sig] = {
          id: sig,
          columns: [],
          unique_values: col.unique_values,
          config: {
            type: isMultipleSheet ? "split" : "ordinal",
            separator: ";",
            encoding: isMultipleSheet ? "onehot" : "ordinal",
            map: {},
          },
        };
      }
      groupsMap[sig].columns.push(col.name);
    });
    stringCols.forEach((col) => {
      if (ungroupedCols.has(col.name)) {
        groupsMap[`ungrouped_${col.name}`] = {
          id: `ungrouped_${col.name}`,
          columns: [col.name],
          unique_values: col.unique_values,
          isUngrouped: true,
          config: {
            type: isMultipleSheet ? "split" : "ordinal",
            separator: ";",
            encoding: isMultipleSheet ? "onehot" : "ordinal",
            map: {},
          },
        };
      }
    });

    setGroups(Object.values(groupsMap));
  }, [stringCols, isMultipleSheet, ungroupedCols]);

  // Функция разгруппировки
  const ungroup = (groupId, colName) => {
    if (colName) {
      // Разгруппировать конкретную колонку
      setUngroupedCols((prev) => new Set([...prev, colName]));
    } else {
      // Разгруппировать всю группу (все колонки группы)
      const group = groups.find((g) => g.id === groupId);
      if (group) {
        setUngroupedCols((prev) => new Set([...prev, ...group.columns]));
      }
    }
  };
  const updateGroupConfig = (groupId, newConfig) => {
    setGroups((prev) =>
      prev.map((g) =>
        g.id === groupId ? { ...g, config: { ...g.config, ...newConfig } } : g,
      ),
    );
  };
  // Состояние конфигурации: { colName: { type: "ordinal"|"one_hot"|"skip", map: {}, separator: "," } }

  // Формирование финального конфига для отправки (убираем служебное поле unique)
  const handleApply = () => {
    const columnsConfig = {};

    groups.forEach((group) => {
      const { config, columns: groupColumns, isUngrouped } = group;

      groupColumns.forEach((colName) => {
        if (config.type === "skip") {
          columnsConfig[colName] = { type: "skip" };
        } else {
          columnsConfig[colName] = {
            type: config.type,
            separator: config.separator || ";",
            encoding: config.encoding || "onehot",
            imputation: "mode",
            ...(config.type === "ordinal" &&
              !isMultipleSheet &&
              !isUngrouped && {
                map: config.map || autoOrdinalMap(group.unique_values || []),
              }),
            ...(config.type === "ordinal" &&
              isUngrouped && {
                map: config.map || autoOrdinalMap(group.unique_values || []),
              }),
          };
        }
      });
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

      {groups.map((group) => (
        <div key={group.id} className="mapper-col">
          <div className="mapper-header">
            <label>
              <b>
                📁 {group.columns.length === 1 ? "Колонка" : "Группа"} (
                {group.columns.length})
              </b>
              <span className="muted" style={{ marginLeft: 8, fontSize: 12 }}>
                {group.columns.join(", ")}
              </span>
            </label>
            <div style={{ display: "flex", gap: 8 }}>
              {group.columns.length > 1 && (
                <button
                  onClick={() => ungroup(group.id)}
                  style={{ fontSize: 12 }}
                >
                  🔓 Разгруппировать всё
                </button>
              )}
              <select
                value={group.config.type}
                onChange={(e) =>
                  updateGroupConfig(group.id, { type: e.target.value })
                }
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
          </div>

          {/* UI для Split */}
          {group.config.type === "split" && (
            <div className="split-controls">
              <label>Разделитель: </label>
              <select
                value={group.config.separator}
                onChange={(e) =>
                  updateGroupConfig(group.id, { separator: e.target.value })
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

          {/* UI для Ordinal */}
          {group.config.type === "ordinal" && !isMultipleSheet && (
            <div className="ordinal-grid">
              <p
                className="muted"
                style={{ gridColumn: "1 / -1", margin: "0 0 6px 0" }}
              >
                {group.columns.length > 1
                  ? "Настройки применяются ко всем колонкам группы"
                  : "Оставьте поля пустыми для авто-шкалы"}
              </p>
              {group.unique_values?.map((val) => (
                <div key={val} className="ordinal-item">
                  <span>{String(val)}</span>
                  <span>=</span>
                  <input
                    type="number"
                    value={group.config.map?.[val] ?? ""}
                    placeholder="0"
                    onChange={(e) =>
                      updateGroupConfig(group.id, {
                        map: { ...group.config.map, [val]: e.target.value },
                      })
                    }
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      ))}

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
        .mapper-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 12px;
          flex-wrap: wrap;
          margin-bottom: 10px;
        }
        .ordinal-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          gap: 8px;
          margin-top: 8px;
        }
        .ordinal-item {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
        }
        .ordinal-item input {
          width: 60px;
          padding: 4px;
          background: var(--bg);
          color: var(--text);
          border: 1px solid var(--border);
          border-radius: 4px;
        }
        .split-controls {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 14px;
          margin-top: 8px;
        }
      `}</style>
    </div>
  );
}
