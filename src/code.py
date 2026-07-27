import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import optuna
import shap

base_dir = Path(__file__).resolve().parent  # resolve делает из относительного пути абсолютный
data_path = base_dir.parent / "data" / "HRdata.csv"
df = pd.read_csv(data_path)

target = "Attrition"
categorical_features = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus",  "OverTime"] # (строковые признаки)

# мы выяснили что есть пара признаков которые не несут полезной информации для обучения модели, например EmployeeNumber(айди работника), Over18, EmployeeCount
df = df.drop(columns=["EmployeeNumber", "EmployeeCount", "Over18"]) # удаляем

X = df.drop(columns="Attrition") # убираем из обучения ответы
y = df["Attrition"]

print("CatBoost:\n")

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # снова включаем сюда stratify чтобы сохранить соотношение(уволившихся оч мало)

# импорт кетбуст
model_cat = CatBoostClassifier(verbose=0, random_state=42, auto_class_weights="Balanced") # verbose отображает тренировочный процесс
model_cat.fit(X_train_cat, y_train_cat, cat_features=categorical_features)

pred_cat = model_cat.predict(X_test_cat) # делаем собственно предикт
accuracy_cat = accuracy_score(y_test_cat, pred_cat) # смотрим на точность предикта
print(accuracy_cat) # accuracy 0.85 это мало о чем говорит потому что если модель просто будет везде говорить что "не уволился" то получит эти 0.85 

print(classification_report(y_test_cat, pred_cat)) # recall низкий. модель боится говорить "уволился" и в итоге находит очень мало реально уволившихся

print(confusion_matrix(y_test_cat, pred_cat))

importance = pd.DataFrame({"Feature": X.columns, "Importance": model_cat.get_feature_importance()})
importance = importance.sort_values(by="Importance", ascending=False)

print(importance.head(15))

btravel_attrition = pd.crosstab(df["BusinessTravel"], df["Attrition"], normalize="index")
print(btravel_attrition) # по данной таблице мы делаем вывод, что уволившиеся чаще были в коммандировках

# план сравнить кетбуст с LOGISTIC REGRESSION
print("Logistic Regression:\n")
df_lr = df.copy()
# т.к. в кетбуст нам давал категориальные признаки, логистическая регрессия так не может, придется использовать onehotencoding get_dummies
df_lr = pd.get_dummies(df_lr, drop_first=True) # станет и больше признаков теперь

X = df_lr.drop(columns="Attrition_Yes")
y = df_lr["Attrition_Yes"]

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scaler нужно добавить так как ему лог. регрессии сложно сравнивать разные величины
scaler = preprocessing.StandardScaler().fit(X_train_lr)
X_train_scaled_lr = scaler.transform(X_train_lr)
X_test_scaled_lr = scaler.transform(X_test_lr)

model_lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000) # class_weight = balanced поможет бороться с зашуганной моделью(низкий recall)
model_lr.fit(X_train_scaled_lr, y_train_lr)

pred_lr = model_lr.predict(X_test_scaled_lr)
accurace_lr = accuracy_score(y_test_lr, pred_lr)
print(accurace_lr)

print(classification_report(y_test_lr, pred_lr))
print(confusion_matrix(y_test_lr, pred_lr))

# roc auc score
prob_cat = model_cat.predict_proba(X_test_cat)[:, 1] # выбираем второй столбик("Yes" уволились. их вероятность)
prob_lr = model_lr.predict_proba(X_test_scaled_lr)[:, 1]

roc_auc_cat = roc_auc_score(y_test_cat, prob_cat)
roc_auc_lr = roc_auc_score(y_test_lr, prob_lr)

print("roc_auc_cat", roc_auc_cat)
print("roc_auc_lr", roc_auc_lr)

# небольшой вывод: кетбуст сильно проигрывает по recall(15%) но чутка впереди по остальным параметрам. auto_class_weights = "Balanced" исправил это (+32%)
# depth у кетбуста по умолчанию 6, я поставил 4, recall еще вырос до 60%, но чутка упал accuracy.
# learning_rate там же тоже можно подкрутить, по умолчанию 0.05, я поставил 0.1 и accuracy вырос на 1%, минимальные прибавки.
# лучшая версия модели кетбуст - до изменения depth и learning_rate (значительная прибавка в recall и незначительная потеря precision)
# ручной перебор параметров делать не буду, т.к. впоследствии планирую использовать Optuna - сама будет искать лучшую комбинацию параметров

# optuna. умно ищет лучшую комбинацию признаков (в отличие от gridsearchcv который перебирает всё)

def objective(trial):
    depth = trial.suggest_int("depth", 4, 10) # берет Случайное число от 4 до 10 и пляшет от него

    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)

    iterations = trial.suggest_int("iterations", 100, 1000)

    model_cat = CatBoostClassifier(depth=depth, learning_rate=learning_rate, iterations=iterations, auto_class_weights="Balanced", random_state=42, verbose=0)

    model_cat.fit(X_train_cat, y_train_cat, cat_features=categorical_features) # categorical_features это все строковые признаки если забыл

    pred = model_cat.predict(X_test_cat)

    score = f1_score(y_test_cat, pred, pos_label = "Yes") # оптуне нужно замаксить f1_score для класса Yes(уволился). f1_score мы взяли из библиотеки .metrics, можно взять другое

    return score

study = optuna.create_study(direction="maximize") # как можно больше
study.optimize(objective, n_trials=30) # 30 раз он будет перебирать depth, learning_rate, iterations ища лучший результат f1_score

print(study.best_params) # на выходе получим 3 параметра в лучшей комбинации
print(study.best_value) # и лучший f1_score

# shap - объяснитель
explainer = shap.TreeExplainer(model_cat) # для кетбуста выбираем TreeExplainer
# считаем shap
shap_values = explainer.shap_values(X_test_cat) # вычислит вклад каждого признака у каждого сотрудника
# красивый график
shap.summary_plot(shap_values, X_test_cat) # красные точки это большие значения признака(большой возраст), синие - маленькие(низкая зарплата)
# если точки слева - отрицательное влияние на нужный признак(уволился_Yes), справа - значит уволился больше. красная точка слева - большая зарплата отриц. влияет на увольнение

# график для одного сотрудника
shap.waterfall_plot(explainer(X_test_cat.iloc[[0]])[0]) # здесь f(x) = 0 -> 50% шанс(увольнения), f(x) = 2 -> 88%; -2 = 12% шанс. f(x) - это сумма всех красных и синих значений на графике.
# в нашем случае f(x) = 0.078 значит что шанс увольнения 52%, чуть выше нуля(50%^)
# лучшие параметры модели и f1_score по мнению optuna: {'depth': 5, 'learning_rate': 0.07077297425903874, 'iterations': 173}; 0.5591397849462365 