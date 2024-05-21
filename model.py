import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

def categorical_features(df):
    cols = ['Marque', 'Modele','Energie','Transmission','Couleur']
    label_encoders = {}
    for col in cols:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])
        # Stocker le label encoder dans le dictionnaire avec le nom de la colonne comme clé
        label_encoders[col] = label_encoder
    return df, label_encoders

df = pd.read_csv("voiture.csv",sep=";")
df, label_encoders = categorical_features(df)

y = df['Prix']
X = df.drop('Prix', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

model = RandomForestRegressor(max_depth=20, min_samples_leaf=1, min_samples_split=10, n_estimators=50, random_state=10)
model.fit(X_train, y_train)

# Prédire les données de test
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, squared=False)
rmse = (mse)**(1/2)
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Sauvegarder le modèle
pickle.dump(model, open('model.pkl', 'wb'))

# Sauvegarder les label encoders
pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))