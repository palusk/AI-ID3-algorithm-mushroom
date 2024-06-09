import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Wczytywanie danych
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]
data = pd.read_csv(url, header=None, names=columns)

# Wstępna analiza danych
print(data.head())
print(data.info())

# Wizualizacja danych
sns.countplot(x='class', data=data)
plt.show()

# Przekształcanie danych na format numeryczny
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Przykłady kodowania etykiet
class_mapping = dict(zip(label_encoders['class'].classes_,
                         label_encoders['class'].transform(label_encoders['class'].classes_)))

habitat_mapping = dict(zip(label_encoders['habitat'].classes_,
                           label_encoders['habitat'].transform(label_encoders['habitat'].classes_)))

cap_shape_mapping = dict(zip(label_encoders['cap-shape'].classes_,
                             label_encoders['cap-shape'].transform(label_encoders['cap-shape'].classes_)))

print("Class mapping (class):", class_mapping)
print("Habitat mapping (habitat):", habitat_mapping)
print("Cap shape mapping (cap-shape):", cap_shape_mapping)

# Podział na cechy i etykiety
X = data.drop('class', axis=1)
y = data['class']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#################################
#       DRZEWO DECYZYJNE        #
#################################
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Wygenerowanie drzewa decyzyjnego
tree_rules = export_text(dt, feature_names=list(X.columns))
print(tree_rules)

# Macierz pomyłek - Drzewo decyzyjne
cm = confusion_matrix(y_test, y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Decision Tree')
plt.show()


#################################
#         LASY LOSOWE           #
#################################
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Ocena ważności cech
importances = rf.feature_importances_
importances_percent = importances * 100  # Przekształcenie na procenty
indices = np.argsort(importances)[::-1]

# Wizualizacja ważności cech w procentach
plt.figure(figsize=(14, 8))  # Zwiększenie rozmiaru wykresu
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances_percent[indices], align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45, ha='right')
plt.xlim([-1, X.shape[1]])
plt.ylabel('Importance (%)')  # Etykieta osi Y w procentach
plt.tight_layout()  # Zapewnia, że wszystkie etykiety są w pełni widoczne
plt.show()

# Macierz pomyłek - Lasy losowe
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Random Forest')
plt.show()

#################################
#     REGRESJA LOGISTYCZNA      #
#################################
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Macierz pomyłek - Regresja logistyczna
cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

#################################
#           METRYKI             #
#################################
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))



#################################
#          GRID SEARCH          #
#################################

# Przeszukiwanie siatki dla drzew decyzyjnych
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=5)
grid_search_dt.fit(X_train, y_train)

# Sortowanie wyników według najlepszych wyników walidacji krzyżowej
results_dt = pd.DataFrame(grid_search_dt.cv_results_)
top_10_params_dt = results_dt.nlargest(10, 'mean_test_score')

print("Top 10 parameters for Decision Tree:")
print(top_10_params_dt[['params', 'mean_test_score']].to_string(index=False))

# Wyciągnięcie domyślnych parametrów dla drzewa decyzyjnego
params = DecisionTreeClassifier().get_params()

selected_params = {key: params[key] for key in ['max_depth', 'min_samples_split']}
print(selected_params)


# Przeszukiwanie siatki dla lasów losowych
param_grid_rf = {
    'n_estimators': [10, 20, 30],
    'max_depth': [None, 5, 10, 15],
    'min_samples_leaf': [5, 10, 15]
}
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

# Sortowanie wyników według najlepszych wyników walidacji krzyżowej
results_rf = pd.DataFrame(grid_search_rf.cv_results_)
top_10_params_rf = results_rf.nlargest(10, 'mean_test_score')

print("Top 10 parameters for Random Forest:")
print(top_10_params_rf[['params', 'mean_test_score']].to_string(index=False))

# Wyciągnięcie domyślnych parametrów dla lasów losowych
params = RandomForestClassifier().get_params()

selected_params = {key: params[key] for key in ['n_estimators', 'max_depth', 'min_samples_leaf']}
print(selected_params)

# Przeszukiwanie siatki dla regresji logistycznej
param_grid_lr = {
    'penalty': ['l1', 'elasticnet'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [1000, 2000, 3000]
}

grid_search_lr = GridSearchCV(lr, param_grid_lr, cv=5)
grid_search_lr.fit(X_train, y_train)

# Sortowanie wyników według najlepszych wyników walidacji krzyżowej
results_lr = pd.DataFrame(grid_search_lr.cv_results_)
top_10_params_lr = results_lr.nlargest(10, 'mean_test_score')

print("Top 10 parameters for Logistic Regression:")
print(top_10_params_lr[['params', 'mean_test_score']].to_string(index=False))


# Wyciągnięcie domyślnych parametrów dla regresji logistycznej
params = LogisticRegression().get_params()

selected_params = {key: params[key] for key in ['penalty', 'C', 'solver', 'max_iter']}
print(selected_params)
