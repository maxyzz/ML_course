import numpy as np

def detect_outliers(x: np.ndarray) -> np.ndarray:
    # 1. Находим 25-й (Q1) и 75-й (Q3) процентили
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    
    # 2. Считаем межквартильный размах (IQR)
    iqr = q3 - q1
    
    # 3. Вычисляем допустимые границы для "нормальных" данных
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 4. Возвращаем True для тех элементов, которые вышли за границы
    # Оператор | означает "ИЛИ"
    return (x < lower_bound) | (x > upper_bound)

# Представим, что это оценки по тесту в классе.
# Большинство учеников получили баллы от 60 до 85,
# но есть пара подозрительно низких и высоких значений.

scores = np.array([70, 75, 80, 65, 83, 78, 74, 72, 68, 69, 150, 20])

outliers_mask = detect_outliers(scores)

print("Оценки:", scores)
print("Маска выбросов:", outliers_mask)

# Для удобства выведем сами выбросы
print("\nНайденные выбросы:")
print(scores[outliers_mask])

### Output:

# Оценки: [ 70  75  80  65  83  78  74  72  68  69 150  20]
# Маска выбросов: [False False False False False False False False False False  True  True]

# Найденные выбросы:
# [150  20]