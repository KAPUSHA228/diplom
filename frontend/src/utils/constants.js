// Служебные колонки (PII) — исключаем из превью и выбора целевой переменной
export const EXCLUDE_COLS = new Set([
  "user",
  "user_id",
  "vk_id",
  "vk id",
  "vk",
  "фамилия",
  "имя",
  "отчество",
  "вуз",
  "факультет",
  "группа",
  "курс",
  "пол",
  "возраст",
  "дата",
  "date",
  "направление подготовки",
  "_source_sheet",
  "_sheet_type",
]);

// Навигационные ссылки для вкладок
export const NAV = [
  { path: "/", label: "Главное" },
  { path: "/timeseries", label: "Временные ряды" },
  { path: "/drift", label: "Дрейф" },
  { path: "/combinations", label: "Комбинации" },
  { path: "/composite", label: "Композитные оценки" },
  { path: "/crosstab", label: "Кросс-таблицы" },
  { path: "/subset", label: "Подмножество" },
  { path: "/imputation", label: "Пропуски" },
  { path: "/experiments", label: "Эксперименты" },
];
