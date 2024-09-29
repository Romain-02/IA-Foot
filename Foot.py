import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
import json
import os


URL = "https://www.footmercato.net/"

dicoChampionnat = {"france" : "france/ligue-1/", "angleterre" : "angleterre/premier-league/", "espagne" : "espagne/liga/", "italie" : "italie/serie-a/", "allemagne" : "allemagne/bundesliga/", "france2" : "france/ligue-2/"}

def trouverJournees(championnat, annee):
    codeS = requests.get(URL + dicoChampionnat[championnat] + annee + "/resultat/").text
    journeesURL = []
    
    i = 1
    
    while(codeS.find("journee-" + str(i) + "\"")  != -1):
        fin = codeS.find("journee-" + str(i)) + len("journee-" + str(i))
        deb = codeS.find("https", fin-150, fin)
        i +=1
        journeesURL.append(codeS[deb:fin])
        
    return journeesURL
        
def recupMatchJournee(journeeURL):
    codeS = requests.get(journeeURL).text
    matchsEquipe = {}
    
    i = codeS.find("matchTeam__name")
    codeS = codeS[i : ]
    
    while(i != -1):
        equipe1 = codeS[2+len("matchTeam__name") : codeS.find("<")]
        i = codeS.find("matchFull__score")
        codeS = codeS[i : ]
        score1 = codeS[codeS.find(">")+1 : codeS.find("<")]
        
        i = codeS[1:].find("matchTeam__name")
        codeS = codeS[i+1 : ]
        
        equipe2 = codeS[2+len("matchTeam__name") : codeS.find("<")]
        i = codeS.find("matchFull__score")
        codeS = codeS[i : ]
        score2 = codeS[codeS.find(">")+1 : codeS.find("<")]
        
        matchsEquipe[equipe1] = (equipe2, score1, score2, 1)
        matchsEquipe[equipe2] = (equipe1, score2, score1, 0)
        
        i = codeS[1:].find("matchTeam__name")
        codeS = codeS[i+1: ]
    return matchsEquipe
        
def actualisePoints(matchsEquipe, pointsChampionnat, saisonEquipe):
    for key in matchsEquipe.keys():
        if matchsEquipe[key][0] in pointsChampionnat:
            if matchsEquipe[key][1] > matchsEquipe[key][2]:
                pointsChampionnat[key] += 3
                saisonEquipe[key].append(1)
            elif matchsEquipe[key][1] == matchsEquipe[key][2]:
                pointsChampionnat[key] += 1
                saisonEquipe[key].append(0.5)
            else:
                saisonEquipe[key].append(0)
    return pointsChampionnat, saisonEquipe

def initialiseChampionnat(championnat, annee):
    codeS = requests.get(URL + dicoChampionnat[championnat] + annee + "/classement/").text
    pointsChampionnat = {}
    saisonEquipe = {}
    
    i = 0
    i = codeS.find("rankingTable__team")
    codeS = codeS[i : ]
    i = codeS.find("alt")
    codeS = codeS[i : ]
    
    while i != -1:
        pointsChampionnat[codeS[5 : 5+codeS[5:].find("\"")]] = 0
        saisonEquipe[codeS[5 : 5+codeS[5:].find("\"")]] = []
        
        i = codeS[1:].find("alt")
        codeS = codeS[i+1 : ]
    return pointsChampionnat, saisonEquipe

def addDataJournee(matchsEquipe, data, pointsChampionnat, saisonEquipe):
    for key in matchsEquipe.keys():
        p1 = pointsChampionnat[key]
        p2 = pointsChampionnat[matchsEquipe[key][0]]
        
        if p1 == 0 and p2 == 0:
            d = [0.5]
        else:
            d = [(p1-p2) / (max(p1, p2)*2) + 0.5]

        for i in range(len(saisonEquipe[key]), 6):
            d.append(0)
            
        i = 1
        while len(d) < 6:
            d.append(saisonEquipe[key][-i-1])
            i += 1
        d.append(matchsEquipe[key][3])
        
        data.append(d)
    return data

def addResultatDefeat(data, m):
    for key in m.keys():
        if m[key][1] > m[key][2] or m[key][1] == m[key][2]:
            data.append([1])
        else:
            data.append([0])
    return data

def addResultatVicotry(data, m):
    for key in m.keys():
        if m[key][1] > m[key][2]:
            data.append([1])
        else:
            data.append([0])
    return data

def addResultatNull(data, m):
    for key in m.keys():
        if m[key][1] > m[key][2]:
            data.append([1])
        elif  m[key][1] == m[key][2]:
            data.append([0.5])
        else:
            data.append([0])
    return data
        
#limit est un booléen qui indique si on commence à partir de la 5ième journée ou si on on commence à partir de la première
def getDataChampionnat(championnat, limit = True, victory = True):
    ANNEE_DEB = 2005
    ANNEE_FIN = 2023
    X = []
    y = []
    for i in range(ANNEE_DEB, ANNEE_FIN):
        if i != 2019:
            annee = str(i) + "-" + str(i+1)
            X, y = getDataChampionnatAnnee(championnat, annee, limit, victory, X, y)
        
    return X, y

def getDataChampionnatAnnee(championnat, annee, limit = True, victory = True, X = [], y = [], jourLimit = 38, deb = 0):
    pointsChampionnat, saisonEquipe = initialiseChampionnat(championnat, annee)
    urls = trouverJournees(championnat, annee) 
    deb = 6 if limit and deb == 0 else deb
    
    for i in range(min(len(urls), jourLimit)):
        m = recupMatchJournee(urls[i])
        pointsChampionnat, saisonEquipe = actualisePoints(m, pointsChampionnat, saisonEquipe)
        if i >= deb:
            X = addDataJournee(m, X, pointsChampionnat, saisonEquipe)
            if victory:
                y = addResultatVicotry(y, m)
            else:
                y = addResultatDefeat(y, m)
    #print(pointsChampionnat, saisonEquipe)
    return X, y

def getStatChampionnat(championnat, annee):
    cpt = 0
    cptV = 0
    cptD = 0
    cptN = 0
    
    urls = trouverJournees(championnat, annee) 
    for url in urls:
        m = recupMatchJournee(url)
        for key in m.keys():
            if m[key][1] > m[key][2]:
                cptV += 1
            elif m[key][1] == m[key][2]:
                cptN += 1
            else:
                cptD += 1
            cpt += 1
    return [cptV / cpt, cptN / cpt, cptD / cpt]

#sauvegarder et importer des données sur les championnats
def sauvegardeTab(tab, fic):
    repertoire = os.path.dirname(fic)
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)
    with open(fic, "w") as fichier:
        json.dump(tab, fichier)
        
def sauvegardeParam(dico, fic):
    for key in dico.keys():
        dico[key] = dico[key].tolist()
    sauvegardeTab(dico, fic)
        
def sauvegardeChampionnat(championnat):
    ANNEE_DEB = 2005
    ANNEE_FIN = 2024
    dataChampionnat = [] #tab constitué de 3 tab de data (resultatVictoire, resultatNull et resultatDéfaite)
    yChampionnat = [[], [], []]
    for i in range(ANNEE_DEB, ANNEE_FIN):
        if i != 2019:
            annee = str(i) + "-" + str(i+1)
            data = []
            y = [[], [], []]
            urls = trouverJournees(championnat, annee)
            pointsChampionnat, saisonEquipe = initialiseChampionnat(championnat, annee)
            for i in range(len(urls)):
                m = recupMatchJournee(urls[i])
                data = addDataJournee(m, data, pointsChampionnat, saisonEquipe)
                y[0] = addResultatVicotry(y[0], m)
                y[1] = addResultatNull(y[1], m)
                y[2] = addResultatDefeat(y[2], m)
                
                # Enregistrer le tableau dans le fichier JSON
                sauvegardeTab(m, "data/" + dicoChampionnat[championnat] + annee + "/journee" + str(i+1) + "/" + "matchs.json")
                sauvegardeTab(saisonEquipe, "data/" + dicoChampionnat[championnat] + annee + "/journee" + str(i+1) + "/" + "saison.json")
                sauvegardeTab(pointsChampionnat, "data/" + dicoChampionnat[championnat] + annee + "/journee" + str(i+1) + "/" + "points.json")
                sauvegardeTab(data, "data/" + dicoChampionnat[championnat] + annee + "/journee" + str(i+1) + "/" + "data.json")
                sauvegardeTab(y[0], "data/" + dicoChampionnat[championnat] + annee + "/journee" + str(i+1) + "/" + "/yvict.json")
                sauvegardeTab(y[1], "data/" + dicoChampionnat[championnat] + annee + "/journee" + str(i+1) + "/" + "/ynull.json")
                sauvegardeTab(y[2], "data/" + dicoChampionnat[championnat] + annee + "/journee" + str(i+1) + "/" + "/ydef.json")
                pointsChampionnat, saisonEquipe = actualisePoints(m, pointsChampionnat, saisonEquipe)
                
            sauvegardeTab(saisonEquipe, "data/" + dicoChampionnat[championnat] + annee + "/fin" + "/" + "saison.json")
            sauvegardeTab(pointsChampionnat, "data/" + dicoChampionnat[championnat] + annee + "/fin" + "/" + "points.json")
            sauvegardeTab(data, "data/" + dicoChampionnat[championnat] + annee + "/fin" + "/" + "data.json")
            sauvegardeTab(y[0], "data/" + dicoChampionnat[championnat] + annee + "/fin" + "/yvict.json")
            sauvegardeTab(y[1], "data/" + dicoChampionnat[championnat] + annee + "/fin" + "/ynull.json")
            sauvegardeTab(y[2], "data/" + dicoChampionnat[championnat] + annee + "/fin" + "/ydef.json")
            for d in data:
                dataChampionnat.append(d)
            for d in y[0]:
                yChampionnat[0].append(d)
            for d in y[1]:
                yChampionnat[1].append(d)
            for d in y[2]:
                yChampionnat[2].append(d)
    sauvegardeTab(dataChampionnat, "data/" + dicoChampionnat[championnat] + "/X.json")
    sauvegardeTab(dataChampionnat, "data/" + dicoChampionnat[championnat] + "/yvict.json")
    sauvegardeTab(dataChampionnat, "data/" + dicoChampionnat[championnat] + "/ynull.json")
    sauvegardeTab(dataChampionnat, "data/" + dicoChampionnat[championnat] + "/ydef.json")


def recupTab(fic):
    with open(fic, "r") as fichier:
        tableau_recupere = json.load(fichier)

    return tableau_recupere

def recupParam(fic):
    tab = recupTab(fic)
    dico = {}
    for cle, valeur in tab.items():
        if isinstance(valeur, list):
            dico[cle] = np.array(valeur)
        else:
            dico[cle] = valeur

    return dico
    
    
    
def initialisation(dimensions):
    
    parametres = {}
    C = len(dimensions)

    np.random.seed(1)

    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parametres

def forward_propagation(X, parametres):
  
  activations = {'A0': X}

  C = len(parametres) // 2

  for c in range(1, C + 1):

    Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
    activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

  return activations

def back_propagation(y, parametres, activations):

  m = y.shape[1]
  C = len(parametres) // 2

  dZ = activations['A' + str(C)] - y
  gradients = {}

  for c in reversed(range(1, C + 1)):
    gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
    gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    if c > 1:
      dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

  return gradients

def update(gradients, parametres, learning_rate):

    C = len(parametres) // 2

    for c in range(1, C + 1):
        parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]

    return parametres

def predict(X, parametres):
  activations = forward_propagation(X, parametres)
  C = len(parametres) // 2
  Af = activations['A' + str(C)]
  return Af >= 0.5

def predictResultat(X, p1, p2):
    activations = forward_propagation(X, p1)
    C = len(p1) // 2
    Af1 = activations['A' + str(C)]
    
    activations = forward_propagation(X, p2)
    C = len(p2) // 2
    Af2 = activations['A' + str(C)]
    
    for i in range(len(Af1)):
        Af1[i] = 1 if Af1[i]>Af2[i]-Af1[i] and Af2[i]>1-Af2[i] else 0
        Af1[i] = 0.5 if Af2[i]-Af1[i]>1-Af2[i] else Af1[i]
    
    return Af1

def getProbaResultat(X, p1, p2):
    activations = forward_propagation(X, p1)

    C = len(p1) // 2
    Af1 = activations['A' + str(C)]
    
    activations = forward_propagation(X, p2)
    C = len(p2) // 2
    Af2 = activations['A' + str(C)]
    
    proba = []
    
    for i in range(len(Af1)):
        proba.append([Af1[i], Af2[i]-Af1[i], 1-Af2[i]])

    return proba

def getPrecision(X, y, p1, p2):
    cpt = 0
    for i in range(len(X)):
        cpt += predictResultat(np.array([X[i]]).T, p1, p2) == y[i]
        
    return cpt / len(X)
    

def deep_neural_network(X_train, y_train, X_test, y_test, hidden_layers = (16, 16, 16), learning_rate = 0.001, n_iter = 3000):
    
    # initialisation parametres
    dimensions = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    np.random.seed(1)
    parametres = initialisation(dimensions)

    # tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(n_iter), 2))
    testing_history = np.zeros((int(n_iter), 2))

    C = len(parametres) // 2

    # gradient descent
    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(y_train, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        training_history[i, 0] = (log_loss(y_train.flatten(), Af.flatten()))

        y_pred = predict(X_train, parametres)
        training_history[i, 1] = (accuracy_score(y_train.flatten(), y_pred.flatten()))
        
        y_pred = predict(X_test, parametres)
        testing_history[i, 1] = (accuracy_score(y_test.flatten(), y_pred.flatten()))

    # Plot courbe d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.plot(testing_history[:, 1], label='test acc')
    plt.legend()
    plt.show()

    return training_history, parametres

def train():
    X_trainV, y_trainV = getDataChampionnat("france", True, True)
    X_trainV, y_trainV = np.array(X_trainV), np.array(y_trainV)
    #on teste notre IA par rapport au championnat quelconque de 2018-2019 pour voir si l'IA est efficace sur un cas général et non sur les données sur lesquelles elle s'est entrainée
    X_testV, y_testV = getDataChampionnatAnnee("angleterre", "2018-2019", True, True, X=[], y=[])
    X_testV, y_testV = np.array(X_testV), np.array(y_testV)

    #on entraine également notre IA à repérer les défaites et non uniquement les victoires pour déterminer ensuite la probabilité du match nul
    X_trainD, y_trainD = getDataChampionnat("france", True, False)
    X_trainD, y_trainD = np.array(X_trainD), np.array(y_trainD)
    X_testD, y_testD = getDataChampionnatAnnee("angleterre", "2018-2019", True, False, X=[], y=[])
    X_testD, y_testD = np.array(X_testD), np.array(y_testD)

    X_trainV = X_trainV.T
    y_trainV = y_trainV.T
    X_testV = X_testV.T
    y_testV = y_testV.T

    X_trainD = X_trainD.T
    y_trainD = y_trainD.T
    X_testD = X_testD.T
    y_testD = y_testD.T

    parametresV = deep_neural_network(X_trainV, y_trainV, X_testV, y_testV, hidden_layers=(16, 16, 16), learning_rate=0.1, n_iter=1500)[1]
    parametresD = deep_neural_network(X_trainD, y_trainD, X_testD, y_testD, hidden_layers=(16, 16, 16), learning_rate=0.1, n_iter=1500)[1]

    sauvegardeParam(parametresV, "/data/ligue-1/parametresV.json")
    sauvegardeParam(parametresD, "/data/ligue-1/parametresD.json")

"""
X_trainV, y_trainV = getDataChampionnat("france", True, True)
X_trainV, y_trainV = np.array(X_trainV), np.array(y_trainV)
X_testV, y_testV = getDataChampionnatAnnee("angleterre", "2018-2019", True, True, X = [], y = [])
X_testV, y_testV = np.array(X_testV), np.array(y_testV)

X_trainD, y_trainD = getDataChampionnat("france", True, False)
X_trainD, y_trainD = np.array(X_trainD), np.array(y_trainD)
X_testD, y_testD = getDataChampionnatAnnee("angleterre", "2018-2019", True, False, X = [], y = [])
X_testD, y_testD = np.array(X_testD), np.array(y_testD)

X_trainV=X_trainV.T
y_trainV=y_trainV.T
X_testV=X_testV.T
y_testV=y_testV.T

X_trainD=X_trainD.T
y_trainD=y_trainD.T
X_testD=X_testD.T
y_testD=y_testD.T

parametresV = deep_neural_network(X_trainV, y_trainV, X_testV, y_testV, hidden_layers = (16, 16, 16), learning_rate = 0.1, n_iter = 1500)[1]
parametresD = deep_neural_network(X_trainD, y_trainD, X_testD, y_testD, hidden_layers = (16, 16, 16), learning_rate = 0.1, n_iter = 1500)[1]

sauvegardeParam(parametresV, "/data/france/ligue-1/parametresV.json")
sauvegardeParam(parametresD, "/data/france/ligue-1/parametresD.json")



Test en pratique avec des vrais côtes (avant de me rendre compte que les côtes
sont calculé en fonction de la probabilité de victoire de sorte à ce que le parieur 
soit toujours légèrement perdant lorsque l'on calcule l'espérance de son gain )
tabCote = [
    [2.55, 3.30, 2.85, 2],
    [1.46, 4.20, 8.00, 0],
    [2.45, 3.50, 2.80, 1],
    [3.80, 3.45, 2.00, 1],
    [3.00, 3.00, 2.60, 1],
    [3.05, 3.35, 2.35, 1],
    [2.00, 3.35, 3.90, 1],
    [3.45, 3.55, 2.10, 2],
    [1.43, 5.10, 7.25, 1],
    
    [6.10, 4.50, 1.52, 2],
    [1.52, 4.30, 6.25, 0],
    [3.60, 3.40, 2.10, 2],
    [1.56, 4.20, 5.90, 1],
    [1.42, 4.80, 7.25, 0],
    [1.84, 3.60, 4.50, 2],
    [2.05, 3.20, 4.00, 2],
    [2.80, 2.90, 2.90, 1],
    [2.90, 3.30, 2.55, 0],
    
    [1.42, 4.50, 8.00, 1],
    [1.64, 4.00, 5.30, 0],
    [1.82, 3.65, 4.40, 0],
    [3.75, 3.25, 2.10, 2],
    [2.10, 3.45, 3.60, 2],
    [4.10, 3.70, 1.88, 0],
    [3.70, 3.35, 2.10, 1],
    [2.85, 3.35, 2.55, 2],
    [2.35, 3.30, 3.15, 2],
    
    [2.70, 3.10, 2.70, 0],
    [1.48, 4.20, 7.00, 0],
    [7.25, 4.80, 1.48, 2],
    [1.68, 3.75, 5.10, 2],
    [1.58, 4.10, 6.00, 0],
    [1.60, 4.10, 5.30, 2],
    [1.62, 3.85, 5.50, 0],
    [2.20, 3.25, 3.40, 1],
    [2.50, 3.30, 3.00, 0],
    
    [4.30, 3.40, 1.95, 2],
    [2.85, 3.15, 2.65, 2],
    [2.75, 3.15, 2.70, 2],
    [2.30, 3.50, 3.00, 2],
    [4.00, 3.35, 2.00, 0],
    [1.42, 4.60, 8.00, 1],
    [3.45, 3.25, 2.20, 2],
    [1.33, 5.50, 9.00, 1],
    [1.70, 3.90, 5.00, 0],
    
    [3.50, 4.00, 2.00, 1],
    [2.85, 3.30, 2.50, 2],
    [5.30, 3.95, 1.64, 2],
    [3.35, 3.20, 2.30, 0],
    [1.86, 3.30, 4.80, 2],
    [2.00, 3.65, 3.65, 1],
    [1.58, 3.90, 6.25, 0],
    [1.47, 4.50, 6.75, 2],
    [2.65, 3.30, 2.80, 2]

]

X, y = getDataChampionnatAnnee("france", "2023-2024", X = [], y = [], jourLimit=24, deb = 18)
print(X)

parametresV, parametresD = recupParam("./data/france/ligue-1/parametresV.json"), recupParam("./data/france/ligue-1/parametresD.json")

def predireMatchs(X, parametresV, parametresD):
    dicoParis = {0 : "Victoire", 1 : "Null", 2 : "Défaite"}
    j = 0
    benef = 0
    paris = 0

    bonParis = 0
    nbParis = 0
    for i in range(0, len(X), 2):
        esp = 0
        print("i : ", i)
        proba = getProbaResultat(np.array([X[i]]).T, parametresV, parametresD)
        print("Probabilités : Victoire : ", proba[0][0], ", Null : ", proba[0][1], ", Défaite : ", proba[0][2])
        print("Côtes : Victoire : ", tabCote[j][0], ", Null : ", tabCote[j][1], ", Défaite : ", tabCote[j][2])
        print("Espérances : Victoire : ", proba[0][0] * tabCote[j][0], ", Null : ", proba[0][1] * tabCote[j][1], ", Défaite : ", proba[0][2] * tabCote[j][2])
        for k in range(3):
            proba[0][k] *= tabCote[-j][k]
        for k in range(3):
            if esp < tabCote[j][k] * proba[0][k]:
                paris = k
                esp = tabCote[j][k] * proba[0][k]
        if paris == tabCote[j][3]:
            benef += tabCote[j][paris]
            bonParis += 1
            
        print("Paris : ", dicoParis[paris], " Paris Gagnant : ", dicoParis[tabCote[j][3]])
        print()
        benef -= 1
        nbParis += 1
        j += 1
        print("Benef : ", benef)
    print("bonParis : ", bonParis, "/", nbParis)
    
for i in range(19, 25):
    m = recupTab("./data/france/ligue-1/2023-2024/journee" + str(i) + "/matchs" + ".json")
    for key in m.keys():
        print(key, " VS ", m[key][0])
    
predireMatchs(X, parametresV, parametresD)
1
"""

#print(getPrecision(X, y, parametresV, parametresD))

def printChoice():
    print("0 - Quitter")
    print("1 - Entrainer l'IA")
    print("2 - Prédire un match")

def getInput(message):
    print(message)
    while (True):
        try:
            return float(input())
        except ValueError:
            print('Donner un nombre')

choix = 1
while(choix != 0):
    printChoice()
    choix = getInput('Donner un nombre entre 0 et 2')
    if choix == 1:
        print("L'IA est entrainé sur les matchs du championnat français sur plus de 20 saisons puis testé sur un autre championnat, les résultats donnés sont donc fiables\nLa récupération des données ainsi que l'entrainement peut prendre un certain temps")
        train()
    elif choix == 2:
        print("Attention, vous ne pouvez prédire efficacement que lorsque la saison a déjà atteint son septième match !\nDans le cas contraire les données tel que le classement ne seront pas représentative")
        X = []
        nbPointsE1 = getInput('Donner le nombre de point de la première équipe')
        nbPointsE2 = getInput('Donner le nombre de point de la deuxième équipe')
        X.append([(nbPointsE1 - nbPointsE2) / (2 * max(nbPointsE1, nbPointsE2)) + 0.5])
        for i in range(6):
            X.append([getInput('Donner le résultat du match n - ' + str(i) + ', 1 pour une victoire, -1 pour une défaite et 0 pour un nul')])
            
        parametresV, parametresD = recupParam("./data/ligue-1/parametresV.json"), recupParam("./data/ligue-1/parametresD.json")
        prediction = getProbaResultat(np.array(X), parametresV, parametresD)
        print('Victoire : ' + str(prediction[0][0]) + ', Nul : ' + str(prediction[0][1]) + ', Défaite : ' + str(prediction[0][2]))


