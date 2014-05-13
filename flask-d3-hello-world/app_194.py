"""
CS194 Final Project
"""
import json
import math
import flask
from flask import request
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sklearn
from jinja2 import Template
from sklearn import linear_model
import random

app = flask.Flask(__name__)

"""
Data Paths
"""
DATA_PATH = "/Users/tbrown126/Documents/cs194/project/cs194_project/data/"
CATEGORY_SEASONPLAYER = "SeasonPlayer/"
CATEGORY_SALARIES= "Salaries/"
CATEGORY_PLAYERCAREER = "PlayerCareerPerGame/"
CATEGORY_GAMES = "Games/"
CATEGORY_GAMEDETAILS = "GameDetails/"

#Modify the two app routes below to return the correct page
@app.route("/hello")
def index():
    """
    When you request the root path, you'll get the index.html template.

    """
    return flask.render_template("index.html")

@app.route("/")
def gindex():
    """
    When you request the gaus path, you'll get the gaus.html template.

    """
    return flask.render_template("index.html")

@app.route("/graph")
def graphRender():
    return flask.render_template("graph.html")

@app.route("/plot")
def getPlot():
    return flask.render_template("plot.html")

@app.route("/predict")
def getPredict():
    return flask.render_template("predictor.html")

@app.route("/predictStats/<name>/<stat>/")
def predictStats(name, stat):
    m1, m2, m3 = train_stat_model(1, stat)
    feat1, feat2, feat3, resp, season = get_features_response(name, stat, 1)

    predicted1 = m1[0].predict(feat1)[:,0].tolist()
    predicted2 = m2[0].predict(feat2)[:,0].tolist()
    predicted3 = m3[0].predict(feat3)[:,0].tolist()

    modelInfo = {}
    modelInfo[0] = [m1[1], m1[2], m1[3]]
    modelInfo[1] = [m2[1], m2[2], m2[3]]
    modelInfo[2] = [m3[1], m3[2], m3[3]]
    
    
    for i in range (0,2):
        predicted1.insert(0,predicted1[i])
    for i in range (0,2):
        predicted3.insert(0,predicted3[i])
    predicted2.insert(0,predicted2[0])

    return json.dumps({"seasons": season.tolist(), "real": resp.tolist(), "predicted1": predicted1, "predicted2": predicted2, "predicted3": predicted3, "modelInfo": modelInfo})

@app.route("/cluster/<int:season>/<pos>/<expVar1>/<expVar2>/<int:clusterNum>")
def getClusterResults(season, pos, expVar1, expVar2, clusterNum):
    """
    On request, this returns a list of players for the given season
    with the relevant x, y, and cluster, as well as cluster points.

    :param season <int>:
        Season to select

    :param pos <string>:
        Position to subset

    :param expVar1 <string>:
        Explanatory variable 1 (x)

    :param expVar2 <string>:
        Explanatory variable 2 (y)

    :returns results:
        A JSON string of players and clusters with their individual
        attributes.

    """
    regularHeaders = seasonDataRegular[season].columns.values.tolist()
    rawDF = rawDF = pd.DataFrame({"Player": seasonDataRegular[season]["Player"], "Tm": seasonDataRegular[season]["Tm"], 
    	"Pos": seasonDataRegular[season]["Pos"], "Age": seasonDataRegular[season]["Age"], "Salary": seasonDataRegular[season]["Salary"]})

    #Get expVar1
    if(expVar1 in regularHeaders):
        rawDF[expVar1] = seasonDataRegular[season][expVar1]
    else:
        rawDF[expVar1] = seasonDataAdvanced[season][expVar1]

    #Get expVar2
    if(expVar2 in regularHeaders):
        rawDF[expVar2] = seasonDataRegular[season][expVar2]
    else:
        rawDF[expVar2] = seasonDataAdvanced[season][expVar2]

    #Subset dataframe to get just the columns and rows we want
    playerDF = rawDF[rawDF["Pos"] == pos].drop("Pos", 1)
    playerDF = playerDF[pd.notnull(playerDF[expVar1])]
    playerDF = playerDF[pd.notnull(playerDF[expVar2])]
    clusterMatrix = playerDF[[expVar1, expVar2]].as_matrix()

    #Cluster on the two explanatory variables
    kmeans_5 = KMeans(n_clusters=clusterNum, n_init=1)
    kmeans_5.fit(clusterMatrix)

    #Do some cleaning
    playerDF["Cluster"] = kmeans_5.labels_.tolist()
    playerDF.reset_index(inplace = True, drop=True)
    centroids = kmeans_5.cluster_centers_

    clusterRadius = {}
    clusterSalaryAverage = {}
    clusterAgeAverage = {}

    for i in range(clusterNum):
    	playerCluster = playerDF[playerDF["Cluster"] == i]
    	numPlayers = len(playerCluster)

    	majorLength = playerCluster[expVar1].std() * .9**(-1 * math.log(numPlayers, 2))
    	minorLength = playerCluster[expVar2].std() * .9**(-1 * math.log(numPlayers, 2))
    	clusterRadius[i] = (majorLength + np.asscalar(centroids[i, 0]), minorLength + np.asscalar(centroids[i, 1]))

    	clusterSalaryAverage[i] = playerCluster["Salary"].mean()
    	clusterAgeAverage[i] = playerCluster["Age"].astype('float').mean()

    #Create the result json object
	playerObj = [{"_name": playerDF.ix[i, "Player"], "_tm": playerDF.ix[i, "Tm"], "age": playerDF.ix[i, "Age"],
    	"salary": playerDF.ix[i,"Salary"], "var1": playerDF.ix[i, 4], "var2": playerDF.ix[i, 5], 
    	"cluster": np.asscalar(playerDF.ix[i, "Cluster"])}
        for i in range(len(playerDF))]
    
    def toDecimal(num):
        return math.floor(num * 100) / 100
        
    def toSalary(num):
        temp = toDecimal(num)
        return '${:,.2f}'.format(temp)
        
    clusterObj = [{"_cluster": i, "farX": clusterRadius[i][0], "farY": clusterRadius[i][1], 
    	"var1": toDecimal(np.asscalar(centroids[i, 0])), "var2": toDecimal(np.asscalar(centroids[i, 1])), "age": toDecimal(clusterAgeAverage[i]),
    	"salary": toSalary(clusterSalaryAverage[i])}
        for i in range(len(centroids))]

    return json.dumps({"_playerObj": playerObj, "_clusterObj": clusterObj})

@app.route("/salary/<int:season>")
def getPlayerSalaries(season):
    """
    On request, this returns a list of players for the given season
    with their respective salary for that season.

    :param season <int>:
        Season to select

    :returns results:
        A JSON string of players and their salaries.
    """
    salaries = seasonSalaries[season]

    salaryObj = [{"_name": salaries.ix[i, 0], "salary": salaries.ix[i, 1]}
        for i in range(len(salaries))]

    return json.dumps(salaryObj)

@app.route("/age/<int:season>")
def getPlayerAge(season):
    """
    On request, this returns a list of players for the given season
    with their respective age for that season.

    :param season <int>:
        Season to select

    :returns results:
        A JSON string of players and their ages.
    """
    rawAges = seasonDataRegular[season]
    ages = rawAges[["Player", "Age"]]

    ageObj = [{"_name": ages.ix[i, 0], "age": ages.ix[i, 1]}
        for i in range(len(ages))]

    return json.dumps(ageObj)

def loadData():
    global seasonDataRegular
    seasonDataRegular = dict()
    global seasonDataAdvanced
    seasonDataAdvanced = dict()
    global seasonSalaries
    seasonSalaries = dict()

    pathToSeason = DATA_PATH + CATEGORY_SEASONPLAYER
    for i in range(9, 14):
        season = i + 2000
        seasonDataRegular[season] = pd.read_csv(pathToSeason + "Regular" + str(season) + "PlusSalary.csv")
        seasonDataAdvanced[season] = pd.read_csv(pathToSeason + "SeasonPlayer" + str(season) + "Advanced.csv")
        seasonSalaries[season] = pd.read_csv(DATA_PATH + CATEGORY_SALARIES + "salaryData" + str(season) + ".csv", header = None)

    players = pd.read_csv(DATA_PATH+"PlayerCareerPerGame/playerPerGame.csv")
    players_adv = pd.read_csv(DATA_PATH+"PlayerCareerPerGame/cleanadv.csv")
    players_clean = players.drop(['Tm', 'Lg', 'Pos'], axis=1)
    global players_combined
    players_combined = players_clean.join(players_adv.drop(['Name', 'Season', 'Age', 'Lg','Pos','G', 'MP','Tm'], axis=1))
    global player_dict
    player_dict = {}
    player_groups = players_combined.groupby('Name')
    for name, group in player_groups:
        seasons = []
        for index, row in group.groupby('Season').agg(np.mean).iterrows():
            seasons.append(row)
        player_dict[name] = seasons

def residuals(features, y, model):
    return y-model.predict(features)

def error_func(features, y, model):
    res = residuals(features, y, model)[0]
    rmse = math.sqrt(np.mean(res**2))
    avgerr = np.mean(abs(res))
    return (rmse, avgerr)

def fit_model(X, y):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return model

def score_model(model, X, y, Xv, yv):
    score1 = model.score(X,y)
    score2 = model.score(Xv, yv)
    return (score1, score2)

def fit_model_and_score(data, response, validation, val_response):
    model = fit_model(data, response)
    return score_model(model, data, response, validation, val_response)

def train_stat_model(option, target):
    index = -1
    for col in players_combined.columns:
        index+=1
        if (col == target):
            break
    target_index = index - 2

    features = {}
    response = {}
    if (True):
        for key, value in player_dict.iteritems():
            if (len(value) >= 4):
                for i in range(2,len(value)-1):
                    new_indices1 = []
                    for val in value[i-1].index:
                        new_indices1.append(val+'1')
                    temp1 = pd.Series(value[i-1].values, index=new_indices1)
                    new_indices2 = []
                    for val in value[i-2].index:
                        new_indices2.append(val+'2')
                    temp2 = pd.Series(value[i-2].values, index=new_indices2)
                    features[key+str(i)] = pd.concat([value[i], temp1, temp2])
                    response[key+str(i)] = value[i+1][target_index]

    feature_df = pd.DataFrame(list(features.values()))
    feature_df.reset_index(inplace = True, drop=True)

    response_df = pd.DataFrame(list(response.values()))
    response_df.reset_index(inplace = True, drop=True)

    #Function 3: sample data
    amount = int(round(len(features) * 0.1))
    rows = random.sample(feature_df.index, amount)
    testing_data = feature_df.ix[rows]
    training_data = feature_df.drop(rows)

    testing_response = response_df.ix[rows]
    training_response = response_df.drop(rows)

    scores1 = fit_model_and_score(training_data, training_response, testing_data, testing_response)

    model1 = fit_model(training_data, training_response)
    rsme1, avgerr1 = error_func(testing_data, testing_response, model1)
        
    if (True):
        for key, value in player_dict.iteritems():
            for i in range(0,len(value)-1):
                features[key+str(i)] = value[i]
                response[key+str(i)] = value[i+1][target_index]

    feature_df = pd.DataFrame(list(features.values()))
    feature_df.reset_index(inplace = True, drop=True)

    response_df = pd.DataFrame(list(response.values()))
    response_df.reset_index(inplace = True, drop=True)

    #Function 3: sample data
    amount = int(round(len(features) * 0.2))
    rows = random.sample(feature_df.index, amount)
    testing_data = feature_df.ix[rows]
    training_data = feature_df.drop(rows)

    testing_response = response_df.ix[rows]
    training_response = response_df.drop(rows)

    scores2 = fit_model_and_score(training_data, training_response, testing_data, testing_response)

    model2 = fit_model(training_data, training_response)
    rsme2, avgerr2 = error_func(testing_data, testing_response, model2)
            
    if (True):
        for key, value in player_dict.iteritems():
            if (len(value) >= 4):
                for i in range(2,len(value)-1):
                    features[key+str(i)] = (value[i]*.55) + (value[i-1]*.3) + (value[i-2]*.15)
                    response[key+str(i)] = value[i+1][target_index]
            
    feature_df = pd.DataFrame(list(features.values()))
    feature_df.reset_index(inplace = True, drop=True)

    response_df = pd.DataFrame(list(response.values()))
    response_df.reset_index(inplace = True, drop=True)

    #Function 3: sample data
    amount = int(round(len(features) * 0.1))
    rows = random.sample(feature_df.index, amount)
    testing_data = feature_df.ix[rows]
    training_data = feature_df.drop(rows)

    testing_response = response_df.ix[rows]
    training_response = response_df.drop(rows)

    scores3 = fit_model_and_score(training_data, training_response, testing_data, testing_response)

    model3 = fit_model(training_data, training_response)
    rsme3, avgerr3 = error_func(testing_data, testing_response, model3)

    modelInfo1 = (model1, avgerr1, rsme1, scores1[0], scores1[1])
    modelInfo2 = (model2, avgerr2, rsme2, scores2[0], scores2[1])
    modelInfo3 = (model3, avgerr3, rsme3, scores3[0], scores3[1])

    return (modelInfo1, modelInfo2, modelInfo3)

def get_features_response(playerName, target, option):
    index = -1
    for col in players_combined.columns:
        index+=1
        if (col == target):
            break
    target_index = index - 2
    player_stats = players_combined[players_combined['Name'] == playerName]

    seasons = player_stats['Season']
    player_stats = player_stats.drop(['Name', 'Season'], axis=1)
    actual_response = player_stats[target]
    key = playerName
    value = []
    for index, row in player_stats.iterrows():
        value.append(row)
    features = {}
    if (True):
        if (len(value) >= 4):
            for i in range(2,len(value)):
                new_indices1 = []
                for val in value[i-1].index:
                    new_indices1.append(val+'1')
                temp1 = pd.Series(value[i-1].values, index=new_indices1)
                new_indices2 = []
                for val in value[i-2].index:
                    new_indices2.append(val+'2')
                temp2 = pd.Series(value[i-2].values, index=new_indices2)
                features[key+str(i)] = pd.concat([value[i], temp1, temp2])
    
    feature_df = pd.DataFrame(list(features.values()))
    feature_df.reset_index(inplace = True, drop=True)
    feature1 = feature_df

    if (True):
        for i in range(0,len(value)):
            features[key+str(i)] = value[i]

    feature_df = pd.DataFrame(list(features.values()))
    feature_df.reset_index(inplace = True, drop=True)
    feature2 = feature_df
            
    if (True):
        if (len(value) >= 4):
            for i in range(2,len(value)):
                features[key+str(i)] = (value[i] + value[i-1] + value[i-2]) *1.0 / 3

    feature_df = pd.DataFrame(list(features.values()))
    feature_df.reset_index(inplace = True, drop=True)

    feature3 = feature_df

    return feature1, feature2, feature3, actual_response, seasons

if __name__ == "__main__":
    import os

    port = 8000

    # Load data
    loadData()

    # Open a web browser pointing at the app.
    os.system("open http://localhost:{0}/".format(port))

    # Set up the development server on port 8000.
    app.debug = True
    app.run(port=port)
