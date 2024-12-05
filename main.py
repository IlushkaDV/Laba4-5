import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Путь к исходному набору данных
file_path = "spaceship_titanic.csv"
data_frame = pd.read_csv(file_path)

# Вывод общей информации о данных
print("Информация о данных:")
print(data_frame.info())
print("\nПример данных:")
print(data_frame.head())

# Удаляем ненужные столбцы
data_frame.drop(["PassengerId", "Name", "Cabin"], axis=1, inplace=True)

# Заполнение пропусков в данных и преобразование типов
data_frame["CryoSleep"] = data_frame["CryoSleep"].fillna(False).astype(bool)
for column in ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
    data_frame[column] = data_frame[column].fillna(data_frame[column].median())

# Преобразование категориальных признаков в числовые one-hot
data_frame = pd.get_dummies(data_frame, columns=["HomePlanet", "Destination"], drop_first=True)

# Сохранение обработанных данных
processed_file_path = "обработанные_данные_spaceship_titanic.csv"
data_frame.to_csv(processed_file_path, index=False, encoding="utf-8")
print(f"Обработанные данные сохранены в файл: {processed_file_path}")

# Разделение данных на признаки и целевую переменную
features = data_frame.drop("Transported", axis=1)
target = data_frame["Transported"].astype(int)

# Разделение данных на обучающую и тестовую выборки
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2,
                                                                            random_state=42)

# Масштабирование данных для улучшения производительности моделей
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Создание и обучение модели Random Forest
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(features_train, target_train)
rf_predictions = random_forest_model.predict(features_test)

# Создание и обучение модели XGBoost
xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
xgboost_model.fit(features_train, target_train)
xgb_predictions = xgboost_model.predict(features_test)

# Оценка моделей и сохранение отчетов
rf_classification_report = classification_report(target_test, rf_predictions,
                                                 target_names=["Не транспортирован", "Транспортирован"],
                                                 zero_division=0)
xgb_classification_report = classification_report(target_test, xgb_predictions,
                                                  target_names=["Не транспортирован", "Транспортирован"],
                                                  zero_division=0)

results_file_path = "результаты_spaceship_titanic.txt"
with open(results_file_path, "w", encoding="utf-8") as file:
    file.write("Отчет классификации Random Forest:\n")
    file.write(rf_classification_report)
    file.write("\nОтчет классификации XGBoost:\n")
    file.write(xgb_classification_report)

# Визуализация важности признаков для модели Random Forest
feature_importance = random_forest_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(features.columns, feature_importance)
plt.xticks(rotation=45, ha="right")
plt.title("Важность признаков (Random Forest)")
plt.tight_layout()
plt.savefig("важность_признаков_rf.png")
plt.show()

print(f"Результаты сохранены в файл: {results_file_path}")
print("График важности признаков сохранен как: важность_признаков_rf.png")
