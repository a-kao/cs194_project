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
from jinja2 import Template

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
    mux = request.args.get('mux', '')
    muy = request.args.get('muy', '')
    if len(mux)==0: mux="3."
    if len(muy)==0: muy="3."
    return flask.render_template("gaus.html",mux=mux,muy=muy)

@app.route("/plot")
def getPlot():
    return flask.render_template("plot.html")

@app.route("/cluster/<int:season>/<pos>/<expVar1>/<expVar2>")
def getClusterResults(season, pos, expVar1, expVar2):
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
    rawDF = rawDF = pd.DataFrame({"Player": seasonDataRegular[season]["Player"], "Tm": seasonDataRegular[season]["Tm"], "Pos": seasonDataRegular[season]["Pos"]})

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

    #Subset dataframe to get just the columns we want
    playerDF = rawDF[rawDF["Pos"] == pos].drop("Pos", 1)
    clusterMatrix = playerDF[[expVar1, expVar2]].as_matrix()

    #Cluster on the two explanatory variables
    kmeans_5 = KMeans(n_clusters=5, n_init=1)
    kmeans_5.fit(clusterMatrix)

    #Do some cleaning
    playerDF["Cluster"] = kmeans_5.labels_.tolist()
    playerDF.reset_index(inplace = True, drop=True)
    centroids = kmeans_5.cluster_centers_
    clusterRadius = {}
    for name, group in playerDF.groupby(['Cluster']):
        radius = 0
        coordinate = (0,0)
        for index, row in group.iterrows():
            dist = math.sqrt((float(row[expVar1])-float(centroids[name, 0]))**2 + (float(row[expVar2])-float(centroids[name, 1]))**2)
            if (dist > radius):
                radius = dist
                coordinate = (row[expVar1], row[expVar2])
        clusterRadius[name] = coordinate

    #Create the result json object
    playerObj = [{"_name": playerDF.ix[i, "Player"], "_tm": playerDF.ix[i, "Tm"], "var1": playerDF.ix[i, 2], 
        "var2": playerDF.ix[i, 3], "cluster": np.asscalar(playerDF.ix[i, "Cluster"])}
        for i in range(len(playerDF))]

    clusterObj = [{"_cluster": i, "farX": clusterRadius[i][0], "farY": clusterRadius[i][1], "var1": np.asscalar(centroids[i, 0]), "var2": np.asscalar(centroids[i, 1])}
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
        seasonDataRegular[season] = pd.read_csv(pathToSeason + "SeasonPlayer" + str(season) + "Regular.csv")
        seasonDataAdvanced[season] = pd.read_csv(pathToSeason + "SeasonPlayer" + str(season) + "Advanced.csv")
        seasonSalaries[season] = pd.read_csv(DATA_PATH + CATEGORY_SALARIES + "salaryData" + str(season) + ".csv", header = None)

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
