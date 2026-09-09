import numpy as np
def one_hot_encode(categories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 1. Найти уникальные категории и отсортировать их
    unique_categories = np.unique(categories)
    
    # 2. Создать словарь для маппинга категории → индекс
    category_to_index = {category: i for i, category in enumerate(unique_categories)}

    # 3. Создать нулевую матрицу нужного размера
    one_hot_matrix = np.zeros((len(categories), len(unique_categories)), dtype=int)

    # 4. Заполнить единицами соответствующие позиции
    for i, category in enumerate(categories):
        index = category_to_index[category]
        one_hot_matrix[i, index] = 1
    
    return one_hot_matrix, list(unique_categories)

# У нас есть список городов, в которых живут пользователи.
# Нужно превратить текстовые названия в числовую матрицу.

cities = np.array(["Москва", "СПб", "Казань", "Москва", "Екатеринбург"])

print("Исходные данные:")
print(cities)

encoded, unique_cities = one_hot_encode(cities)

print("\nУникальные категории:")
print(unique_cities)

print("\nРезультат one-hot кодирования:")
print(encoded)

### Output:
# Исходные данные:
# ['Москва' 'СПб' 'Казань' 'Москва' 'Екатеринбург']

# Уникальные категории:
# [np.str_('Екатеринбург'), np.str_('Казань'), np.str_('Москва'), np.str_('СПб')]

# Результат one-hot кодирования:
# [[0 0 1 0]
#  [0 0 0 1]
#  [0 1 0 0]
#  [0 0 1 0]
#  [1 0 0 0]]