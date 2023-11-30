import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
# Função para determinar a categoria
def determine_category(sp_type):
    if isinstance(sp_type, float):
        sp_type = str(sp_type)
    roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    sp_type = sp_type.upper()

    for numeral in roman_numerals:
        if numeral in sp_type:
            if numeral in ['I', 'II', 'III']:
                return 'Giant'
            elif numeral in ['IV', 'V', 'VI', 'VII']:
                return 'Dwarfs'

    return np.nan

# Carregar e preparar os dados
file_path = 'Star_raw_nan_fixed.csv'
data = pd.read_csv(file_path)
data = data.drop(columns=['ID'])
imputer = SimpleImputer(strategy='mean')
data[['Vmag', 'Plx', 'e_Plx', 'B-V']] = imputer.fit_transform(data[['Vmag', 'Plx', 'e_Plx', 'B-V']])

# Categorizar os tipos de estrelas e limpar dados
data['Category'] = data['SpType'].apply(determine_category)
data = data.dropna(subset=['Category'])

# Transformar SpType e Category
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

X = data.drop(columns=['SpType', 'Category'])
y = data['Category']

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encoding das labels
y = to_categorical(y)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Função para criar o modelo MLP
def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Treinar o Modelo MLP
model_mlp = create_model()
model_mlp.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Treinar o Modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, np.argmax(y_train, axis=1))

# Inicializar listas para armazenar as pontuações
scores_mlp = []
scores_knn = []

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, np.argmax(y, axis=1)):
    # Dividir os dados
    X_train_fold, X_test_fold = X[train_index], X[test_index] 
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # Treinar e avaliar o modelo MLP
    model_mlp = create_model()
    model_mlp.fit(X_train_fold, y_train_fold, epochs=100, batch_size=32, verbose=0)
    scores_mlp.append(model_mlp.evaluate(X_test_fold, y_test_fold, verbose=0)[1])

    # Treinar e avaliar o modelo KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_fold, np.argmax(y_train_fold, axis=1))
    y_pred_knn_fold = knn.predict(X_test_fold)
    scores_knn.append(accuracy_score(np.argmax(y_test_fold, axis=1), y_pred_knn_fold))

# Imprimir as médias das acurácias
print(f"Validação Cruzada - Acurácia MLP: {np.mean(scores_mlp):.2f}")
print(f"Validação Cruzada - Acurácia KNN: {np.mean(scores_knn):.2f}")

# Realizar o teste T emparelhado
t_statistic, p_value = ttest_rel(scores_mlp, scores_knn)
print(f"Teste T emparelhado: t = {t_statistic}, p = {p_value}")

# Avaliar métricas específicas (Gigantes e Anãs) para KNN
y_pred_classes_knn = knn.predict(X_test)
print("KNN Classification Report:")
print(classification_report(np.argmax(y_test, axis=1), y_pred_classes_knn, target_names=label_encoder.classes_))

# Avaliar métricas específicas para MLP
y_pred_classes_mlp = model_mlp.predict(X_test)
y_pred_classes_mlp = np.argmax(y_pred_classes_mlp, axis=1)
print("MLP Classification Report:")
print(classification_report(np.argmax(y_test, axis=1), y_pred_classes_mlp, target_names=label_encoder.classes_))