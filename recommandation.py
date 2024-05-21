import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("donnees_voiture.csv", sep=";")
print(df.info())
print(df.head())
df['Prix'] = df['Prix'].astype(str)
df['Annee'] = df['Annee'].astype(str)
df['Kilometrage'] = df['Kilometrage'].astype(str)
df['Nombres de portes'] = df['Nombres de portes'].astype(str)
df['Puissance'] = df['Puissance'].astype(str)
df['Nombre de places'] = df['Nombre de places'].astype(str)
df['Emission de CO2'] = df['Emission de CO2'].astype(str)
df['Consommation de carburant'] = df['Consommation de carburant'].astype(str)


df['Meta'] = df['Marque'] + " " + df['Modele'] + " " + df['Prix'] + " " + df['Annee'] + " "+ df['Kilometrage']+ " " + df['Energie']+ " " + df['Transmission']+ " " + df['Nombres de portes']+ " " + df['Puissance']+ " " +df['Nombre de places']+ " " +df['Emission de CO2']+ " " + df['Consommation de carburant']+ " " + df['Couleur'] + " " + df['Url']
print(df['Meta'])

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Meta'])

print(tfidf_matrix)
