import pandas as pd
import numpy as np
import random
import math
from sklearn import linear_model

import pylab as pl
import matplotlib.pyplot as pyplot

def residuals(features, y, model):
    return y-model.predict(features)

def error_func(features, y, model):
    res = residuals(features, y, model)[0]
    rmse = math.sqrt(np.mean(res**2))
    avgerr = np.mean(res)
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

option = 0
target = 'WS'


DATA_PATH = "/Users/tbrown126/Documents/cs194/project/cs194_project/data/"

players = pd.read_csv(DATA_PATH+"PlayerCareerPerGame/playerPerGame.csv")
players_adv = pd.read_csv(DATA_PATH+"PlayerCareerPerGame/cleanadv.csv")

players_clean = players.drop(['Tm', 'Lg', 'Pos'], axis=1)
players_combined = players_clean.join(players_adv.drop(['Name', 'Season', 'Age', 'Lg','Pos','G', 'MP','Tm'], axis=1))


#returns (mean error, RMSE, training score, testing score)
def train_stat_model(option, target):
    print players_combined[target].describe()
    index = -1
    for col in players_combined.columns:
        index+=1
        if (col == target):
            break
    target_index = index - 2

    #Function 1: convert dataframe to a dictionary
    player_dict = {}
    player_groups = players_combined.groupby('Name')
    for name, group in player_groups:
        seasons = []
        for index, row in group.groupby('Season').agg(np.mean).iterrows():
            seasons.append(row)
        player_dict[name] = seasons


    features = {}
    response = {}
    if (option == 0):
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
        
    if (option == 1):
        for key, value in player_dict.iteritems():
            for i in range(0,len(value)-1):
                features[key+str(i)] = value[i]
                response[key+str(i)] = value[i+1][target_index]
            
    if (option == 2):
        for key, value in player_dict.iteritems():
            if (len(value) >= 4):
                for i in range(2,len(value)-1):
                    features[key+str(i)] = (value[i] + value[i-1] + value[i-2]) *1.0 / 3
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

    scores = fit_model_and_score(training_data, training_response, testing_data, testing_response)

    model = fit_model(training_data, training_response)
    rsme, avgerr = error_func(testing_data, testing_response, model)

    return (avgerr, rsme, scores[0], scores[1])

def get_features_response(playerName, target, option):
    index = -1
    for col in players_combined.columns:
        index+=1
        if (col == target):
            break
    target_index = index - 2
    player_stats = players_combined[players_combined['Name'] == playerName].drop(['Name', 'Season'], axis=1)
    actual_response = player_stats[target]
    key = playerName
    value = []
    for index, row in player_stats.iterrows():
        value.append(row)
    features = {}
    response = {}
    if (option == 0):
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
        
    if (option == 1):
        for i in range(0,len(value)-1):
            features[key+str(i)] = value[i]
            response[key+str(i)] = value[i+1][target_index]
            
    if (option == 2):
        if (len(value) >= 4):
            for i in range(2,len(value)-1):
                features[key+str(i)] = (value[i] + value[i-1] + value[i-2]) *1.0 / 3
                response[key+str(i)] = value[i+1][target_index]
    feature_df = pd.DataFrame(list(features.values()))
    feature_df.reset_index(inplace = True, drop=True)

    response_df = pd.DataFrame(list(response.values()))
    response_df.reset_index(inplace = True, drop=True)
    return feature_df, response_df