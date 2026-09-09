import numpy as np
def normalize(X: np.ndarray) -> np.ndarray:
    min = X.min(axis=0, keepdims=True)
    max = X.max(axis=0, keepdims=True)
    result = (X - min) / (max - min)
    
    return result

# До нормализации
age = np.array([25, 30, 35])
income = np.array([50000, 100000, 150000])
matrix = np.array([[6, 5, 4], [7, 2, 1]])

# После нормализации
age_norm = normalize(age)
income_norm = normalize(income)
matrix_norm = normalize(matrix)

print("Возраст (до нормализации):", age)
print("Возраст (после нормализации):", age_norm)

print("\nДоход (до нормализации):", income)
print("Доход (после нормализации):", income_norm)

print("\nМатрица (до нормализации):", matrix)
print("Матрица (после нормализации):", matrix_norm)

### Output:
# Возраст (до нормализации): [25 30 35]
# Возраст (после нормализации): [0.  0.5 1. ]

# Доход (до нормализации): [ 50000 100000 150000]
# Доход (после нормализации): [0.  0.5 1. ]

# Матрица (до нормализации): [[6 5 4], [7 2 1]]
# Матрица (после нормализации): [[0. 1. 1.], [1. 0. 0.]]