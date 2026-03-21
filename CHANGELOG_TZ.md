# Изменения для соответствия ТЗ

## Что было изменено

### 1. Структура входных данных

**Было:**
- Простые синтетические данные с признаками: `grade`, `attendance`, `tasks_recent`, `stress_score`

**Стало:**
- Данные из 4 источников согласно ТЗ:
  1. **Данные об успеваемости студентов**: `avg_grade`, `grade_std`, `min_grade`, `max_grade`, `n_courses`, `avg_brs`
  2. **Данные по итогам анкетирования**: `satisfaction_score`, `engagement_score`, `workload_perception`
  3. **Данные по итогам психологического тестирования**: `stress_level`, `motivation_score`, `anxiety_score`
  4. **Данные эссе**: `n_essays`, `avg_essay_grade`

### 2. Функция загрузки данных (`load_data`)

**Изменения:**
- Теперь поддерживает загрузку из 4 источников (CSV файлы)
- Генерирует синтетические данные, соответствующие структуре ТЗ
- Агрегирует данные по студентам (соответствует ТЗ 6.1 и 6.2)

**Новые функции:**
- `_generate_grades_data()` - генерация данных об успеваемости
- `_generate_questionnaires_data()` - генерация данных анкетирования
- `_generate_psych_tests_data()` - генерация данных психологического тестирования
- `_generate_essays_data()` - генерация данных эссе
- `_aggregate_student_features()` - агрегация данных по студентам

### 3. Функция создания композитных признаков (`add_composite_features`)

**Новые композитные признаки:**
1. `trend_grades` - тренд успеваемости (разница между семестрами)
2. `grade_stability` - стабильность успеваемости (коэффициент вариации)
3. `cognitive_load` - когнитивная нагрузка (комбинация workload_perception, stress_level, n_essays)
4. `overall_satisfaction` - общая удовлетворенность (из анкетирования)
5. `psychological_wellbeing` - психологическое благополучие (из психологического тестирования)
6. `academic_activity` - академическая активность (комбинация успеваемости и активности)

### 4. Автоматическое определение признаков

**Было:**
- Жестко заданный список признаков

**Стало:**
- Автоматическое определение доступных признаков из данных
- Гибкая система, которая работает с любыми комбинациями источников данных

## Соответствие ТЗ

### ✅ Система представления данных для анализа

- **Интерактивная подсистема сводки и группировки данных по объектам** → `_aggregate_student_features()`, `cluster_students()`
- **Интерактивная подсистема оценки неизвестных значений** → `preprocess_data()` с `fillna()`
- **Подсистема формирования выходных данных** → Экспорт CSV, подготовка для ML

### ✅ Аналитическая система

- **Интерактивная подсистема выборки показателей и объектов группировки** → `select_features()`, автоматическое определение признаков
- **Интерактивная подсистема выбора и реализации метода анализа** → `correlation_analysis()`, `train_and_evaluate()`, `cluster_students()`

## Как использовать

### Загрузка реальных данных

```python
df = load_data(
    grades_path='data/grades.csv',
    questionnaires_path='data/questionnaires.csv',
    psych_tests_path='data/psych_tests.csv',
    essays_path='data/essays.csv',
    generate_synthetic=False
)
```

### Использование синтетических данных (для тестирования)

```python
df = load_data(generate_synthetic=True)  # По умолчанию
```

## Формат CSV файлов

### grades.csv
```csv
student_id,course_name,grade,semester,brs_score
0,Математика,4.5,1,85.0
0,Физика,4.0,1,80.0
...
```

### questionnaires.csv
```csv
student_id,satisfaction_score,engagement_score,workload_perception
0,4.2,3.8,3.5
1,3.1,2.9,4.2
...
```

### psych_tests.csv
```csv
student_id,stress_level,motivation_score,anxiety_score
0,6.5,7.2,5.8
1,8.1,4.3,7.9
...
```

### essays.csv
```csv
student_id,n_essays,avg_essay_grade
0,3,4.0
1,2,3.5
...
```

## Обратная совместимость

Старый код продолжит работать, так как функция `load_data()` по умолчанию генерирует синтетические данные, соответствующие новой структуре ТЗ.

