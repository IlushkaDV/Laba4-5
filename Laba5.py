import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scipy.stats import randint, uniform
import warnings

# Отключаем предупреждения
warnings.filterwarnings('ignore')

data_path = "spaceship_titanic.csv"
df = pd.read_csv(data_path)

# Удаляем ненужные столбцы
df.drop(["PassengerId", "Name", "Cabin"], axis=1, inplace=True)

# Обработка пропущенных значений
# Заполняем пропуски в столбце CryoSleep значением False и преобразуем в логический тип
df["CryoSleep"] = df["CryoSleep"].fillna(False).astype(bool)

# Заполняем пропуски в числовых столбцах медианой
for col in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
    df[col] = df[col].fillna(df[col].median())

# Преобразуем категориальные столбцы в числовые с помощью one-hot кодирования
df = pd.get_dummies(df, columns=["HomePlanet", "Destination"], drop_first=True)

# Разделяем данные на признаки и целевую переменную
X = df.drop("Transported", axis=1)
y = df["Transported"].astype(int)

# Разбиваем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализацуем данные
# Инициализируем стандартизатор
scaler = StandardScaler()
# Обучаем стандартизатор на обучающих данных и преобразуем их
X_train = scaler.fit_transform(X_train)
# Преобразуем тестовые данные с теми же параметрами
X_test = scaler.transform(X_test)

# Модель случайного леса
rf_model = RandomForestClassifier(random_state=42)
# Модель XGBoost
xgb_model = XGBClassifier(eval_metric="logloss", random_state=42)

# Шаг 6: Определение сеток гиперпараметров
# Параметры для Grid Search для Random Forest
rf_param_grid_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Параметры для Random Search для Random Forest
rf_param_grid_random = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4),
    'bootstrap': [True, False]
}

# Параметры для Байсова подхода для Random Forest
rf_param_grid_bayes = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4),
    'bootstrap': Categorical([True, False])
}

# Параметры для Grid Search для XGBoost
xgb_param_grid_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Параметры для Случайный поиск для XGBoost
xgb_param_grid_random = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

# Параметры для Байсова подход для XGBoost
xgb_param_grid_bayes = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.2, prior='uniform'),
    'subsample': Real(0.6, 1.0, prior='uniform'),
    'colsample_bytree': Real(0.6, 1.0, prior='uniform')
}


# Функция для поиска гиперпараметров
def hyperparameter_search(model, param_grid, search_type='grid', n_iter=20, cv=3):
    if search_type == 'grid':
        # Случайный поиск
        search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    elif search_type == 'random':
        # Случайный поиск
        search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, cv=cv,
                                    scoring='accuracy', random_state=42, n_jobs=-1)
    elif search_type == 'bayes':
        # Байсова подход
        search = BayesSearchCV(model, param_grid, n_iter=n_iter, cv=cv, scoring='accuracy',
                               random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unsupported search type. Choose from 'grid', 'random', 'bayes'.")

    # Засекаем время начала поиска
    start_time = time.time()
    # Выполняем поиск
    search.fit(X_train, y_train)
    # Засекаем время окончания поиска
    end_time = time.time()

    # Возвращаем результаты и затраченное время
    return search, end_time - start_time

#  Ищем гиперпараметры
# Словарь с моделями и их параметрами
models = {
    'Random Forest': {
        'model': rf_model,
        'param_grid_grid': rf_param_grid_grid,
        'param_grid_random': rf_param_grid_random,
        'param_grid_bayes': rf_param_grid_bayes
    },
    'XGBoost': {
        'model': xgb_model,
        'param_grid_grid': xgb_param_grid_grid,
        'param_grid_random': xgb_param_grid_random,
        'param_grid_bayes': xgb_param_grid_bayes
    }
}

# Список методов поиска
search_methods = ['grid', 'random', 'bayes']
# Список для хранения результатов
results = []

# Цикл по моделям и методам поиска
for model_name, config in models.items():
    model = config['model']
    for method in search_methods:
        print(f"Поиск гиперпараметров для {model_name} с использованием {method} поиска...")
        param_grid = config[f'param_grid_{method}']
        # Выполняем поиск
        search, duration = hyperparameter_search(model, param_grid, search_type=method, n_iter=20, cv=3)

        # Получаем лучшие параметры и оценку
        best_params = search.best_params_
        best_score = search.best_score_
        # Сохраняем результаты
        results.append({
            'Model': model_name,
            'Search Method': method,
            'Best Params': best_params,
            'Best CV Score': best_score,
            'Time (s)': duration
        })
        print(f"Лучшие параметры: {best_params}")
        print(f"Лучший CV Score: {best_score:.4f}")
        print(f"Время выполнения: {duration:.2f} секунд\n")

# Оцениваем модели
def evaluate_model(model, X_test, y_test):
    # Предсказываем на тестовых данных
    y_pred = model.predict(X_test)
    # Вычисляем точность
    acc = accuracy_score(y_test, y_pred)
    # Получаем отчет по классификации
    report = classification_report(y_test, y_pred)
    return acc, report

# Список для хранения результатов оценки
evaluation_results = []

# Цикл по результатам поиска
for r in results:
    model_name = r['Model']
    search_method = r['Search Method']
    best_params = r['Best Params']

    # Создаем модель с лучшими параметрами
    if model_name == 'Random Forest':
        model = RandomForestClassifier(**best_params, random_state=42)
    elif model_name == 'XGBoost':
        model = XGBClassifier(**best_params, eval_metric="logloss", random_state=42)

    # Обучаем модель
    model.fit(X_train, y_train)
    # Оцениваем модель
    acc, report = evaluate_model(model, X_test, y_test)
    # Сохраняем результаты
    evaluation_results.append({
        'Model': model_name,
        'Search Method': search_method,
        'Test Accuracy': acc,
        'Classification Report': report
    })

# Шаг 10: Вывод результатов
for res in evaluation_results:
    print(f"Модель: {res['Model']} | Метод поиска: {res['Search Method']}")
    print(f"Точность на тестовой выборке: {res['Test Accuracy']:.4f}")
    print(f"Отчет по классификации:\n{res['Classification Report']}\n")


# Преобразуем результаты в DataFrame
results_df = pd.DataFrame(results)
print("Результаты поиска гиперпараметров:")
print(results_df)

# Шаг 11: Анализ результатов и обоснование выбора метода
# Сводная таблица результатов
summary = results_df.groupby(['Model', 'Search Method']).agg({
    'Best CV Score': 'max',
    'Time (s)': 'mean'
}).reset_index()

print("Сводная таблица результатов:")
print(summary)

# Визуализируем результаты
import seaborn as sns
import matplotlib.pyplot as plt

# Настраиваем график
plt.figure(figsize=(12, 6))
sns.scatterplot(data=summary, x='Time (s)', y='Best CV Score', hue='Search Method', style='Model', s=100)
plt.title('Соотношение качество/время для разных методов поиска гиперпараметров')
plt.xlabel('Время (секунды)')
plt.ylabel('Лучший CV Score')
plt.legend(title='Метод поиска', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Определяем лучший метод для каждой модели
for model_name in summary['Model'].unique():
    model_data = summary[summary['Model'] == model_name]
    # Сортируем по лучшему результату и времени
    best_method = model_data.sort_values(by=['Best CV Score', 'Time (s)'], ascending=[False, True]).iloc[0]
    print(f"Для модели {model_name} лучший метод: {best_method['Search Method']}")
    print(f"Лучший CV Score: {best_method['Best CV Score']:.4f}")
    print(f"Время выполнения: {best_method['Time (s)']:.2f} секунд\n")
