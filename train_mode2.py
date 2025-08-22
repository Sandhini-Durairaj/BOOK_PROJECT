# importation des librairies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# importation du jeu donnee et affichage
dt = pd.read_csv('books.csv', delimiter=',',on_bad_lines='skip')
dt.head()

dt.describe()

# type des colonnes
dt.dtypes



masque = dt['average_rating'].astype(str).str.contains(r'[a-zA-Z]', na=False)

# Afficher ces lignes
print(dt[masque][['average_rating', 'authors', 'title']])

#afficher la lgne 3348
dt.loc[3348]
# suppression de la ligne 
dt =dt.drop(index=3348)
dt = dt.drop(index= 4701)
dt = dt.drop(index= 4702)
dt = dt.reset_index(drop=True)

# formatage correcte de chaque ligne
dt['publication_date'] = pd.to_datetime(dt['publication_date'], errors='coerce')
dt['average_rating'] = dt['average_rating'].astype(float)
dt['ratings_count'] = dt['ratings_count'].astype(int)
dt['num_pages'] = dt['  num_pages'].astype(int)
dt['text_reviews_count'] = dt['text_reviews_count'].astype(int)
dt['isbn13'] = dt['isbn13'].astype(str)

# verification des doublons 
dt.duplicated().sum()

sns.boxplot(x=dt['average_rating'])
# Identifier automatiquement les types de variables
target = 'average_rating'

# Colonnes à exclure
cols_to_exclude_num = ['bookID']
cols_to_exclude = ['isbn13', 'isbn']

# Variables numériques 
numeric_cols = dt.select_dtypes(include=[np.number]).columns.drop([target] + [col for col in cols_to_exclude_num if col in dt.columns]).tolist()

# Variables catégorielles 
categorical_cols = [col for col in dt.select_dtypes(include=['object']).columns if col not in cols_to_exclude]

print(f"Variable cible : {target}")
print(f"Variables numériques : {numeric_cols}")
print(f"Variables catégorielles : {categorical_cols}")
print(f"\nTaille du dataset : {dt.shape}")


# Corrélation entre variables numériques et la cible
correlation_with_target = dt[numeric_cols + [target]].corr()[target].drop(target).sort_values(key=abs, ascending=False)

print("Corrélations avec average_rating :")
for var, corr in correlation_with_target.items():
    print(f"{var}: {corr:.4f}")

# Visualisation
plt.figure(figsize=(10, 6))
sns.heatmap(dt[numeric_cols + [target]].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation - Variables numériques')
plt.show()

X = dt.drop(columns=[target])
y = dt[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),           # Normalisation des colonnes numériques
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # Encodage des colonnes catégorielles
    ]
)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RMSE : {np.sqrt(mse):.3f}")
print(f"R² : {r2:.3f}")