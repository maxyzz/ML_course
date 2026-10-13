import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List

class GradientDescentMse:
    """
    Класс для реализации градиентного спуска в задаче линейной регрессии
    с минимизацией среднеквадратичной ошибки (MSE).
    """

    def __init__(
        self,
        samples: pd.DataFrame,
        targets: pd.DataFrame,
        learning_rate: float = 1e-3,
        threshold: float = 1e-6,
        copy: bool = True,
    ):
        """
        Инициализирует модель градиентного спуска для линейной регрессии.

        Args:
            samples (pd.DataFrame):
                Матрица признаков (наблюдения по строкам, признаки по столбцам).

            targets (pd.DataFrame):
                Вектор целевых значений (таргетов), соответствующий наблюдениям.

            learning_rate (float, optional):
                Скорость обучения, для масштабирования градиента при обновлении коэффициентов.
                По умолчанию 1e-3.

            threshold (float, optional):
                Порог изменения функции потерь (MSE), ниже которого выполнение алгоритма прекращается.
                По умолчанию 1e-6.

            copy (bool, optional):
                Если True, создаётся копия матрицы признаков.
                Если False — оригинальная матрица будет использоваться напрямую (in-place).
                По умолчанию True.

        Raises:
            ValueError: Если `samples` или `targets` пусты, или если количество строк не совпадает.
        """
        # Проверка: матрица признаков не должна быть пустой
        if samples.empty:
            raise ValueError("Матрица признаков 'samples' не должна быть пустой.")

        # Проверка: вектор таргетов не должен быть пустым
        if targets.empty:
            raise ValueError("Вектор целевых значений 'targets' не должен быть пустым.")

        # Проверка: количество наблюдений (строк) в samples и targets должно совпадать
        # Каждому объекту должен соответствовать один таргет
        if samples.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Количество строк в 'samples' ({samples.shape[0]}) и 'targets' ({targets.shape[0]}) должно совпадать."
            )

        # При copy=True создаём копию samples, иначе работаем с оригиналом (in-place)
        if copy:
            self.samples = samples.copy()
        else:
            self.samples = samples

        # Сохраняем входные параметры
        self.targets = targets
        self.learning_rate = learning_rate
        self.threshold = threshold

        # Инициализируем вектор коэффициентов beta единицами (можно использовать функцию np.ones()).
        # Размерность должна соответствовать числу признаков (столбцов) в матрице samples.
        self.beta = np.ones(samples.shape[1])

        # Инициализируем пустой словарь для хранения значений функции потерь по итерациям
        self.iteration_loss_dict = {}

    def __str__(self) -> str:
        """
        Возвращает строковое представление текущего состояния модели.

        Returns:
            str:
                Строка с информацией о размере входных данных, параметрах обучения,
                начальных коэффициентах модели и количестве сохранённых итераций.
        """
        return (
            f"GradientDescentMse(\n"
            f"  samples_shape={self.samples.shape},\n"
            f"  targets_shape={self.targets.shape},\n"
            f"  learning_rate={self.learning_rate},\n"
            f"  threshold={self.threshold},\n"
            f"  beta={self.beta},\n"
            f"  iteration_loss_dict={self.iteration_loss_dict}\n)"
        )

    def add_constant_feature(self):
        """
        Добавляет в матрицу признаков столбец 'constant' со значением 1, если он ещё не добавлен.

        Это нужно для учёта свободного члена в линейной регрессии.
        Также вектор коэффициентов beta расширяется одним значением 1 для нового признака.
        """
        # Если столбец `constant`` отсутствует в `self.samples`
        if 'constant' not in self.samples.columns:
            # Добавляем в `samples` константный признак (свободный член) со значением 1
            self.samples['constant'] = 1

            # Расширяем вектор коэффициентов beta на один элемент для нового признака
            self.beta = np.append(self.beta, 1)

    def calculate_gradient(self) -> np.ndarray:
        """
        Вычисляет градиент функции потерь (MSE) по каждому признаку.

        Для каждого признака рассчитывается производная функции потерь,
        которая используется для обновления коэффициентов модели на следующем шаге.

        Returns:
            np.ndarray: Вектор градиентов по всем признакам.
        """
        # Вычисляем предсказания модели: матрица признаков * вектор коэффициентов
        predictions = self.samples.dot(self.beta)

        # Вычисляем вектор ошибок между предсказанными и истинными значениями
        # Размер этого массива равен количеству строк в self.samples
        errors = predictions - self.targets.values.flatten()

        # Инициализируем numpy массив для градиентов (один градиент на каждый признак)
        gradients = np.zeros(self.samples.shape[1])

        # Для каждого признака рассчитываем градиент MSE по формуле: градиент = 2 * mean(error * x)
        for i in range(self.samples.shape[1]):
            # Получаем все значения i-го признака (i-го столбца)
            feature_column = self.samples.iloc[:, i]

            # Вычисляем градиент по этому признаку и сохраняем в соответствующую ячейку
            gradients[i] = 2 * np.mean(errors * feature_column)

        return gradients

    def calculate_mse_loss(self) -> float:
        """
        Вычисляет среднеквадратичную ошибку (MSE) текущей модели.

        Ошибка рассчитывается как среднее значение квадратов разностей между
        предсказанными и истинными значениями.

        Returns:
            float: Текущее значение функции потерь MSE.
        """
        # Считаем предсказания модели: X * β
        predictions = self.samples.dot(self.beta)

        # Вычисляем среднеквадратичную ошибку: MSE = mean((y_pred - y_true)^2)
        mse = np.mean((predictions - self.targets.values.flatten()) ** 2)

        return mse

    def iteration(self):
        """
        Выполняет один шаг градиентного спуска.

        Использует текущий вектор градиента и learning_rate для обновления beta.
        """
        # Вычисляем градиент по всем признакам
        grad = self.calculate_gradient()

        # Обновляем коэффициенты по формуле градиентного спуска: β = β - learning_rate * grad
        self.beta = self.beta - self.learning_rate * grad

    def learn(self, show_progress: bool = True):
        """
        Запускает обучение модели методом градиентного спуска до достижения порогового значения изменения ошибки.

        Алгоритм:
            - Вычисляет начальное значение ошибки (MSE).
            - Итеративно обновляет веса модели.
            - После каждой итерации сохраняет текущее значение ошибки.
            - Останавливается, когда изменение ошибки становится меньше threshold.
            - При show_progress=True отображает индикатор прогресса с помощью tqdm.

        Args:
            show_progress (bool): Отображать ли прогресс обучения. По умолчанию True.
        """
        # Начальное значение ошибки
        curr_loss = self.calculate_mse_loss()
        self.iteration_loss_dict[0] = curr_loss
        step_id = 1

        # Настраиваем прогресс-бар
        progress = tqdm(
            desc="Обучение модели",
            unit="ит",
            dynamic_ncols=True,
            disable=not show_progress,
        )

        while True:
            # Один шаг градиентного спуска
            self.iteration()

            # Вычисляем новую ошибку и сохраняем ее в словарь
            new_loss = self.calculate_mse_loss()
            self.iteration_loss_dict[step_id] = new_loss

            # Обновляем прогресс
            progress.set_postfix(loss=new_loss)
            progress.update(1)

            # Проверяем условие остановки: если абсолютное изменение ошибки меньше порога
            if abs(curr_loss - new_loss) < self.threshold:
                break

            # Переходим к следующей итерации
            curr_loss = new_loss
            step_id += 1

        progress.close()
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Вычисляет предсказания модели для новых объектов на основе текущих коэффициентов (beta).

        Метод автоматически добавляет столбец 'constant', если он использовался
        при обучении, но отсутствует в переданных данных. Также проверяется,
        что число признаков в новых данных совпадает с размерностью вектора beta.

        Args:
            data (pd.DataFrame): Таблица признаков для предсказания (каждая строка — объект).

        Returns:
            np.ndarray: Вектор предсказанных значений, длина которого равна числу объектов в data.

        Raises:
            ValueError: Если число признаков в data не совпадает с размерностью beta.
        """

        # Создаём копию, чтобы не менять переданный DataFrame
        data_copy = data.copy()

        # Проверяем, использовался ли константный признак при обучении модели.
        # Если в обучающих данных был столбец "constant", то он ОБЯЗАТЕЛЬНО должен быть и в новых данных.
        # Если его нет — добавляем.
        if "constant" in self.samples.columns and "constant" not in data_copy.columns:
            # Добавляем столбец "constant" со значением 1 для всех объектов
            data_copy["constant"] = 1

        # Проверяем, совпадает ли число признаков в новых данных с количеством коэффициентов beta.
        # Если нет — выбрасываем исключение. Это нужно, чтобы избежать ошибок при матричном умножении.
        if data_copy.shape[1] != self.beta.shape[0]:
            raise ValueError(
                f"Количество признаков в data ({data_copy.shape[1]}) должно совпадать "
                f"с размерностью beta ({self.beta.shape[0]})."
            )

        # Вычисляем предсказания: y_pred = X * beta
        y_pred = data_copy.dot(self.beta).to_numpy()

        return y_pred
def run_gradient_descent_experiments(
    samples: pd.DataFrame,
    targets: pd.DataFrame,
    threshold_list: List[float],
    lr_list: List[float],
    show_progress: bool = False,
) -> Dict:
    """
    Запускает градиентный спуск для разных комбинаций порогов остановки (threshold)
    и коэффициентов обучения (learning rate).

    Аргументы:
        samples: np.ndarray
            Матрица признаков (объекты x признаки).
        targets: np.ndarray
            Вектор целевых значений.
        threshold_list: list[float]
            Список значений порога сходимости градиентного спуска.
        lr_list: list[float]
            Список значений коэффициента обучения (learning rate).
        show_progress: bool, по умолчанию False
            Если True — выводит информацию о ходе обучения.

    Возвращает:
        dict:
            Словарь вложенной структуры:
            {
                threshold_1: {
                    learning_rate_1: {
                        "losses": {step_1: mse_1, step_2: mse_2, ..., step_n: mse_n},
                        "final_loss": final_mse
                    },
                    learning_rate_2: { ... },
                    ...
                },
                threshold_2: { ... },
                ...
            }

            Где:
                - "losses" — словарь с историей значения MSE по шагам обучения,
                - "final_loss" — итоговое значение MSE после завершения обучения.
    """
    results = {}

    # Перебираем все значения порога остановки
    for t in threshold_list:
        results[t] = {}

        # Перебираем все значения коэффициента обучения
        for lr in lr_list:
            # Инициализация класса градиентного спуска
            GD = GradientDescentMse(
                samples=samples, targets=targets, threshold=t, learning_rate=lr
            )

            # Добавляем константный признак
            GD.add_constant_feature()

            # Запускаем процесс обучения, можно отобразить прогресс по желанию
            GD.learn(show_progress=show_progress)

            # Получаем историю значений функции потерь (MSE) по итерациям из словаря iteration_loss_dict
            losses = GD.iteration_loss_dict.copy()

            # Получаем финальное значение MSE после завершения обучения
            final_loss = losses[max(losses)]

            # Сохраняем результаты в словарь
            results[t][lr] = {"losses": losses, "final_loss": final_loss}

    # Возвращаем все накопленные результаты
    return results

