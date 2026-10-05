import pandas as pd

from gradient_descent_mse import GradientDescentMse, run_gradient_descent_experiments

# 6 task
# X = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
# y = pd.DataFrame([1, 2, 3])

# model = GradientDescentMse(X, y)
# print(model)

# 7 task
# X = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
# y = pd.DataFrame([1, 2, 3])

# model = GradientDescentMse(X, y)

# print(f"Состояние модели ДО добавления константного признака:\n{model}\n")

# model.add_constant_feature()

# print(f"Состояние модели ПОСЛЕ вызова add_constant_feature:\n{model}\n")

# 8 task
# X = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
# y = pd.DataFrame([1, 2, 3])

# model = GradientDescentMse(X, y)

# model.add_constant_feature()

# print("Вектор градиентов по признакам (включая константный):")
# print(model.calculate_gradient())

# 9 task
# X = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
# y = pd.DataFrame([1, 2, 3])

# model = GradientDescentMse(X, y)

# model.add_constant_feature()

# print("Среднеквадратичная ошибка (MSE) при текущих весах модели:")
# print(model.calculate_mse_loss())

# 10 task
# X = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
# y = pd.DataFrame([1, 2, 3])

# model = GradientDescentMse(X, y)

# model.add_constant_feature()

# print(f"Начальные значения beta: {model.beta}")

# model.iteration()

# print(f"Значения beta после одной итерации градиентного спуска: {model.beta}")

# 11 task
# X = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
# y = pd.DataFrame([1, 2, 3])

# model = GradientDescentMse(X, y)

# model.add_constant_feature()

# model.learn()

# print("Обученные коэффициенты beta:", model.beta)


#  12 task
# samples = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
# targets = pd.DataFrame([1, 2, 3])

# threshold_list = [1e-2, 1e-3, 1e-4, 1e-5]
# lr_list = [1e-2, 1e-3, 1e-4, 1e-5]

# results = run_gradient_descent_experiments(samples, targets, threshold_list, lr_list)
# # plot_learning_curves(results)

# # Выведем первый результат из группы экспериментов
# threshold, lr_dict = next(iter(results.items()))
# lr, data = next(iter(lr_dict.items()))

# print(f"Threshold: {threshold:.3e}, Learning rate: {lr:.3e}")
# print("Losses (обрезано):", {k: round(v, 3) for k, v in data["losses"].items()})
# print(f"Final MSE: {data['final_loss']:.3f}")

# task 13
import pandas as pd
import numpy as np

# Генерация данных
np.random.seed(42)
n_total = 1000
n_train = 800

X = pd.DataFrame(
    {"feature1": np.random.rand(n_total), "feature2": np.random.rand(n_total)}
)

true_beta = np.array([3.0, 5.0])
y = pd.DataFrame(
    X["feature1"] * true_beta[0]
    + X["feature2"] * true_beta[1]
    + np.random.normal(0, 0.2, size=n_total)
)

# Разделяем на обучение и тест
X_train = X.iloc[:n_train].reset_index(drop=True)
y_train = y[:n_train]
X_test = X.iloc[n_train:].reset_index(drop=True)
y_test = y[n_train:]

# Обучаем модель
model = GradientDescentMse(X_train, y_train, learning_rate=0.05, threshold=1e-6)
model.add_constant_feature()
model.learn()

# Вывод истинных коэффициентов
print(f"Истинные коэффициенты (true_beta): {true_beta}")

# Итоговые веса (коэффициенты)
print("\nИтоговые веса (beta):")
for name, coef in zip(model.samples.columns, model.beta):
    print(f"  {name}: {coef:.4f}")

# Предсказания на тесте
y_test_pred = model.predict(X_test)

# Первые 5 реальных и предсказанных значений теста
print("\nПервые 5 реальных значений теста:")
print(y_test[:5].values)
print("\nПервые 5 предсказанных значений теста:")
print(y_test_pred[:5])