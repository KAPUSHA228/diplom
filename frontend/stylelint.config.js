export default {
  // Расширяем стандартную конфигурацию (основа)
  extends: [
    "stylelint-config-standard", // стандартные правила CSS[citation:4]
    "stylelint-config-recess-order", // порядок свойств (группировка)[citation:1]
    "stylelint-config-standard-scss", // поддержка SCSS
  ],

  // Плагины для дополнительных возможностей
  plugins: [
    "stylelint-order", // для кастомного порядка свойств
    "stylelint-scss", // правила для SCSS
  ],

  // Игнорируем определённые файлы/папки
  ignoreFiles: [
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/*.min.css",
  ],

  rules: {
    // === Цвета ===
    "color-hex-length": "short", // короткие HEX (#fff вместо #ffffff)[citation:9]
    "color-named": "never", // запрет named colors (red, blue)[citation:1]
    "color-no-hex": null, // запретить HEX цвета

    // === Единицы измерения ===
    "unit-disallowed-list": ["pt", "pc", "in", "cm", "mm"], // запрет типографских единиц
    "unit-allowed-list": [
      "%",
      "deg",
      "px",
      "rem",
      "em",
      "ms",
      "s",
      "fr",
      "vh",
      "vw",
      "rgb",
      "rgba",
      "hsl",
      "hsla",
      "hex",
    ], // разрешённые единицы[citation:1]

    // === Селекторы ===
    "selector-max-id": 0, // запрет ID селекторов[citation:1][citation:2]
    "selector-max-type": 1, // максимум один тип селектора (например, div)
    "selector-max-compound-selectors": 2, // не более 2 составных селекторов (.a .b .c - запрещено)[citation:2]
    "selector-class-pattern": [
      // паттерн именования классов (kebab-case)[citation:2]
      "^[a-z][a-z0-9]*(-[a-z0-9]+)*$",
      { resolveNestedSelectors: true },
    ],
    "selector-pseudo-class-no-unknown": [
      true,
      {
        // запрет неизвестных псевдоклассов
        ignorePseudoClasses: ["global", "local"], // игнорировать для CSS Modules
      },
    ],

    // === Свойства ===
    "declaration-no-important": true, // запрет !important[citation:2]
    "declaration-block-no-redundant-longhand-properties": true, // использовать сокращённые свойства (margin вместо margin-top и т.д.)

    // === Порядок свойств (расширенный) ===
    "order/properties-order": [
      // кастомный порядок свойств[citation:6]
      // Позиционирование
      "position",
      "top",
      "right",
      "bottom",
      "left",
      "z-index",

      // Отображение и поток
      "display",
      "flex-direction",
      "flex-wrap",
      "flex-flow",
      "justify-content",
      "align-items",
      "align-self",
      "align-content",
      "grid-template-columns",
      "grid-template-rows",
      "grid-gap",

      // Размеры
      "width",
      "min-width",
      "max-width",
      "height",
      "min-height",
      "max-height",
      "margin",
      "margin-top",
      "margin-right",
      "margin-bottom",
      "margin-left",
      "padding",
      "padding-top",
      "padding-right",
      "padding-bottom",
      "padding-left",

      // Фоны и границы
      "background",
      "background-color",
      "background-image",
      "background-size",
      "border",
      "border-radius",
      "box-shadow",

      // Текст и шрифты
      "font",
      "font-family",
      "font-size",
      "font-weight",
      "line-height",
      "text-align",
      "text-decoration",
      "text-transform",
      "color",

      // Трансформации и анимации
      "transform",
      "transition",
      "animation",

      // Прочее
      "opacity",
      "visibility",
      "cursor",
      "overflow",
    ],

    // === SCSS специфичные правила ===
    "scss/dollar-variable-pattern": "^[a-z][a-z0-9-]*$", // паттерн для переменных (kebab-case)
    "scss/at-mixin-pattern": "^[a-z][a-z0-9-]*$", // паттерн для миксинов
    "scss/at-extend-no-missing-placeholder": true, // запрет @extend без placeholder (%)
    "scss/load-no-partial-leading-underscore": true, // при @import не указываем _ в начале
    "scss/operator-no-newline-after": true, // операторы без переноса строки после
  },
};
