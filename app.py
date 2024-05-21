from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app=Flask(__name__)

model=pickle.load(open("model.pkl","rb"))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
tfidf_vectorizer = TfidfVectorizer()
df = pd.read_csv("donnees_voiture.csv", sep=";")

def matrice_tfidf():
    df['Prix'] = df['Prix'].astype(str)
    df['Annee'] = df['Annee'].astype(str)
    df['Kilometrage'] = df['Kilometrage'].astype(str)
    df['Nombres de portes'] = df['Nombres de portes'].astype(str)
    df['Puissance'] = df['Puissance'].astype(str)
    df['Nombre de places'] = df['Nombre de places'].astype(str)
    df['Emission de CO2'] = df['Emission de CO2'].astype(str)
    df['Consommation de carburant'] = df['Consommation de carburant'].astype(str)
    df['Meta'] = df['Marque'] + " " + df['Modele'] + " " + df['Prix'] + " " + df['Annee'] + " "+ df['Kilometrage']+ " " + df['Energie']+ " " + df['Transmission']+ " " + df['Nombres de portes']+ " " + df['Puissance']+ " " +df['Nombre de places']+ " " +df['Emission de CO2']+ " " + df['Consommation de carburant']+ " " + df['Couleur'] + " " + df['Url']
    #print(df['Meta'])
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Meta'])
    #print("matrice meta:")
    #print(tfidf_matrix)
    #print("fin")
    return tfidf_matrix

def matrice_tfidf_3(X1,X2,X3):
    df[X1] = df[X1].astype(str)
    df[X2] = df[X2].astype(str)
    df[X3] = df[X3].astype(str)
    df['Meta'] = df[X1] + " " + df[X2] + " " + df[X3] 
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Meta'])
    return tfidf_matrix

def matrice_tfidf_4(X1,X2,X3,X4):
    df[X1] = df[X1].astype(str)
    df[X2] = df[X2].astype(str)
    df[X3] = df[X3].astype(str)
    df[X4] = df[X4].astype(str)
    df['Meta'] = df[X1] + " " + df[X2] + " " + df[X3] +" "+df[X4]
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Meta'])
    return tfidf_matrix

def matrice_tfidf_5(X1,X2,X3,X4,X5):
    df[X1] = df[X1].astype(str)
    df[X2] = df[X2].astype(str)
    df[X3] = df[X3].astype(str)
    df[X4] = df[X4].astype(str)
    df[X5] = df[X5].astype(str)
    df['Meta'] = df[X1] + " " + df[X2] + " " + df[X3] +" "+df[X4] + " "+df[X5]
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Meta'])
    return tfidf_matrix

def matrice_tfidf_6(X1,X2,X3,X4,X5,X6):
    df[X1] = df[X1].astype(str)
    df[X2] = df[X2].astype(str)
    df[X3] = df[X3].astype(str)
    df[X4] = df[X4].astype(str)
    df[X5] = df[X5].astype(str)
    df[X6] = df[X6].astype(str)
    df['Meta'] = df[X1] + " " + df[X2] + " " + df[X3] +" "+df[X4] + " "+df[X5]+ " "+df[X6]
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Meta'])
    return tfidf_matrix

def matrice_tfidf_7(X1,X2,X3,X4,X5,X6,X7):
    df[X1] = df[X1].astype(str)
    df[X2] = df[X2].astype(str)
    df[X3] = df[X3].astype(str)
    df[X4] = df[X4].astype(str)
    df[X5] = df[X5].astype(str)
    df[X6] = df[X6].astype(str)
    df[X7] = df[X7].astype(str)
    df['Meta'] = df[X1] + " " + df[X2] + " " + df[X3] +" "+df[X4] + " "+df[X5]+ " "+df[X6]+ " "+df[X7]
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Meta'])
    return tfidf_matrix

def predict_price(input_data):
    # Assurer que les données sont dans le bon ordre
    input_order = ['Marque', 'Modele', 'Annee', 'Kilometrage', 'Energie', 'Transmission','Portes','Puissance' ,'Places', 'Co2','Carburant', 'Couleur']
    input_data_encoded = []
    #print(input_data)
    for feature in input_order:
        value = input_data[feature]
        if isinstance(value, str):
            label_encoder = label_encoders[feature]
            encoded_value = label_encoder.transform([value])[0]
        else:
            encoded_value = value
        input_data_encoded.append(encoded_value)

    # Faire la prédiction
    predicted_price = model.predict([input_data_encoded])
    return predicted_price[0]

# Exemple d'utilisation
#input_data = {'Marque': 'Dacia', 'Modele': 'Scenic', 'Annee': 2000, 'Kilometrage': 151100, 'Energie': 'GPL ou GNL', 'Transmission': 'Manuelle', 'Portes': 2, 'Puissance': 4, 'Places': 5, 'Co2': 141, 'Carburant': 8.5, 'Couleur': 'Blanc'}
#predicted_price = predict_price(input_data)
#print(predicted_price)

def recuperer_donnes_prediction():
    marque= request.args.get('choix_marque')
    modele= request.args.get('choix_modele')
    carburant = request.args.get('choix_carburant')
    vitesse= request.args.get('choix_vitesse')
    couleur = request.args.get('choix_couleur')
    puissance = int(request.args.get('puissance'))
    kilometrage = int(request.args.get('kilometrage'))
    conso_carburant = float(request.args.get('Conso'))
    emission_co2= float(request.args.get('Emission'))
    portes = int(request.args.get('portes'))
    places = int(request.args.get('places'))
    annee = int(request.args.get('annee'))

    input_data={
        'Marque':marque,
        'Modele':modele,
        'Annee':annee, 
        'Kilometrage':kilometrage,
        'Energie':carburant,
        'Transmission':vitesse,
        'Portes':portes,
        'Puissance':puissance,
        'Places':places,
        'Co2':emission_co2,
        'Carburant':conso_carburant,
        'Couleur':couleur
        }  

    return input_data

@app.route("/")
def Home():
    return render_template("welcome.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == 'GET':
        input_data=recuperer_donnes_prediction()  
    #print(input_data)
        critere = {
            'critere1': request.args.get('critere1'),'critere2': request.args.get('critere2'),'critere3': request.args.get('critere3'),'critere4': request.args.get('critere4'),'critere5': request.args.get('critere5'),'critere6': request.args.get('critere6'),'critere7': request.args.get('critere7')
        }

        valeurs = [valeur for valeur in critere.values() if valeur is not None]
        #print(valeurs)

        if len(valeurs) == 0:
            predicted_price=int(predict_price(input_data))
            urls = None

        if len(valeurs) == 3:
            predicted_price=int(predict_price(input_data))
            input_data['Prix'] = predicted_price 
            tfidf_matrix_3=matrice_tfidf_3(valeurs[0], valeurs[1], valeurs[2])
            input_data2=str(input_data[valeurs[0]])+" "+str(input_data[valeurs[1]])+" "+str(input_data[valeurs[2]])
            input_vector = tfidf_vectorizer.transform([input_data2])
            cosin_sim = cosine_similarity(input_vector, tfidf_matrix_3)
            similar_indices = cosin_sim.argsort()[0][-5:][::-1]  # Get top 5 most similar
            urls = df.loc[similar_indices, 'Url'].tolist()

        if len(valeurs) == 4:
            predicted_price=int(predict_price(input_data))
            input_data['Prix'] = predicted_price 
            tfidf_matrix_4=matrice_tfidf_4(valeurs[0], valeurs[1], valeurs[2],valeurs[3])
            input_data2=str(input_data[valeurs[0]])+" "+str(input_data[valeurs[1]])+" "+str(input_data[valeurs[2]])+" "+str(input_data[valeurs[3]])
            print(input_data2)
            input_vector = tfidf_vectorizer.transform([input_data2])
            cosin_sim = cosine_similarity(input_vector, tfidf_matrix_4)
            similar_indices = cosin_sim.argsort()[0][-5:][::-1]  # Get top 5 most similar
            urls = df.loc[similar_indices, 'Url'].tolist()

        if len(valeurs) == 5:
            predicted_price=int(predict_price(input_data))
            input_data['Prix'] = predicted_price 
            tfidf_matrix_5=matrice_tfidf_5(valeurs[0], valeurs[1], valeurs[2],valeurs[3],valeurs[4])
            input_data2=str(input_data[valeurs[0]])+" "+str(input_data[valeurs[1]])+" "+str(input_data[valeurs[2]])+" "+str(input_data[valeurs[3]])+" "+str(input_data[valeurs[4]])
            input_vector = tfidf_vectorizer.transform([input_data2])
            cosin_sim = cosine_similarity(input_vector, tfidf_matrix_5)
            similar_indices = cosin_sim.argsort()[0][-5:][::-1]  # Get top 5 most similar
            urls = df.loc[similar_indices, 'Url'].tolist()

        if len(valeurs) == 6:
            predicted_price=int(predict_price(input_data))
            input_data['Prix'] = predicted_price 
            tfidf_matrix_6=matrice_tfidf_6(valeurs[0], valeurs[1], valeurs[2],valeurs[3],valeurs[4],valeurs[5])
            input_data2=str(input_data[valeurs[0]])+" "+str(input_data[valeurs[1]])+" "+str(input_data[valeurs[2]])+" "+str(input_data[valeurs[3]])+" "+str(input_data[valeurs[4]])+" "+str(input_data[valeurs[5]])
            input_vector = tfidf_vectorizer.transform([input_data2])
            cosin_sim = cosine_similarity(input_vector, tfidf_matrix_6)
            similar_indices = cosin_sim.argsort()[0][-5:][::-1]  # Get top 5 most similar
            urls = df.loc[similar_indices, 'Url'].tolist()
        
        if len(valeurs) == 7:
            predicted_price=int(predict_price(input_data))
            input_data['Prix'] = predicted_price 
            tfidf_matrix_7=matrice_tfidf_7(valeurs[0], valeurs[1], valeurs[2],valeurs[3],valeurs[4],valeurs[5],valeurs[6])
            input_data2=str(input_data[valeurs[0]])+" "+str(input_data[valeurs[1]])+" "+str(input_data[valeurs[2]])+" "+str(input_data[valeurs[3]])+" "+str(input_data[valeurs[4]])+" "+str(input_data[valeurs[5]])+" "+str(input_data[valeurs[6]])
            input_vector = tfidf_vectorizer.transform([input_data2])
            cosin_sim = cosine_similarity(input_vector, tfidf_matrix_7)
            similar_indices = cosin_sim.argsort()[0][-5:][::-1]  # Get top 5 most similar
            urls = df.loc[similar_indices, 'Url'].tolist()

    return render_template("reponse.html", predicted_price=predicted_price,urls=urls,input_data=input_data,critere=valeurs)



if __name__=="__main__":

    app.run(debug=True)

