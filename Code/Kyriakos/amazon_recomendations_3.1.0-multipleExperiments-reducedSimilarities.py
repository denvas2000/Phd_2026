# configuration
# -------------

#DIRECTORY = r'C:/Users/DS-pc/PycharmProjects/amazon_recomendations/twitter-emotion-recognition-master/'

#DIRECTORY = r'C:/Users/laptop/Desktop/amazon_recomendations/twitter-emotion-recognition-master/'
#DIRECTORY = r'/home/local/costas/classes/ptyx/michailidis/2021-07-23/amazon_recomendations_2.3.2/'
DIRECTORY = r'C:/PHD/Paper_michailidis/Data/reviews_Video_Games_5.json/'

#defaultFile = 'Magazine_Subscriptions.json'
defaultFile = 'Video_Games_5.json'
#defaultFile = 'Musical_Instruments_5.json'
defaultSentimentCalculation = 'anew'

# weights help us to change the impact of a specific emotion or the overall rating for our simulation, for example some emotions
# we can assume that some emotions have bigger impact on out simulation same goes to the overall rating
weights = {'overall': 0.5, 'anger': 0.6, 'disgust': 0.3, 'fear': 0.1, 'joy': 0.6, 'sadness': 0.4, 'surprise': 0.2}

# we need the scaling because our emotion function have fields from [0,1] with sum of 1 that mean its not possible to impliment it
# on the amazon dataset (like most datasets) because the range is [1 ,5] thats why we need the scaling, on differgccent datasets it must change
scalings = {'overall': 0.1, 'anger': 1, 'disgust': 1, 'fear': 1, 'joy': 1, 'sadness': 1, 'surprise': 1}

NUM_PREDICTIONS_TO_COMPUTE = 500

# -------------
# end of configuration


# Imports
import math
import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'theano'
import pandas as pd
import gzip
import numpy as np
from keras import backend as K
from parse import parse
#emotion predictor lib must have it localy
#emotion predictor path in drive the path must be inside the folder
from emotion_predictor import EmotionPredictor
from pandas.io.json import json_normalize
import json
import sys
import random
# end of imports



os.chdir(DIRECTORY)

if (len(sys.argv) >= 2):
    file = sys.argv[1]
else:
    file = defaultFile

fullFileSpec = DIRECTORY + file



if ((len(sys.argv) >= 3) and (sys.argv[2] == 'cached')):
    sentimentCalculation = 'cached'
elif ((len(sys.argv) >= 3) and (sys.argv[2] == 'anew')):
    sentimentCalculation = 'anew'
else:
    sentimentCalculation = defaultSentimentCalculation

# Function to read data from JSON file. Returns a data frame

# read cached JSON file


def readJSONFile(filespec, filetype):
    # if file type = 'anew', we use this code
    if (filetype == 'anew'):
        df0 = pd.read_json(filespec, lines = True)
        df = pd.DataFrame(df0, columns = ['reviewerName','reviewerID','reviewText','overall','asin'])
    else:
        # else, if file type = cached, use this code
        with open(filespec)as json_data:
            data = json.load(json_data)
        df = pd.DataFrame(data['data'])
        df.columns = data['columns']

    return df


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# Similarities


#in this functon we calculate the pearson similarity of all pair of the dataframe
# in reviews1_mean,reviews2_mean we have the mean values of the rating

#Paragraph 2.1.2
def simPFweights(reviews1, reviews2, reviews1_mean, reviews2_mean, weights, scalings):
    numCommon = 0
    nominator = 0
    denom1 = 0
    denom2 = 0
    for index, row in reviews1.iterrows():
        # check if the same item has been rated by user2
        otherUserReview = reviews2[reviews2.asin.eq(row.asin)]
        # check if the same item mean has been rated by user2
        if (len(otherUserReview) > 0):
          # same product rated
          numCommon = numCommon + 1
          #implimitation of the function according to sim-metrics
          for dimension in weights:
              nominator = nominator + (((row[dimension] - reviews1_mean[dimension]) * (otherUserReview.iloc[0][dimension] - reviews2_mean[dimension]))) * weights[dimension] * scalings[dimension]
              denom1 = denom1 + ((row[dimension] - reviews1_mean[dimension]) ** 2) * weights[dimension] * scalings[dimension]
              denom2 = denom2 + ((otherUserReview.iloc[0][dimension] - reviews2_mean[dimension]) ** 2) * weights[dimension] * scalings[dimension]
    try:
      return nominator / (math.sqrt(denom1 * denom2))
    except ZeroDivisionError:
      return 0


# in this functon we calculate the cosine similarity of all pair of the dataframe
def simCFweights(reviews1, reviews2, weights, scalings):
  numCommon = 0
  nominator = 0
  denom1 = 0
  denom2 = 0
  for index, row in reviews1.iterrows():
    # check if the same item has been rated by user2
    otherUserReview = reviews2[reviews2.asin.eq(row.asin)]
    if (len(otherUserReview) > 0):
      # same product rated
      numCommon = numCommon + 1
      for dimension in weights:
        nominator = nominator + row[dimension] * otherUserReview.iloc[0][dimension] * weights[dimension] * scalings[dimension]
        denom1 = denom1 + weights[dimension] * row[dimension] * row[dimension] * scalings[dimension]
        denom2 = denom2 + weights[dimension] * otherUserReview.iloc[0][dimension] * otherUserReview.iloc[0][dimension] * scalings[dimension]
  try:
      return nominator / math.sqrt(denom1) / math.sqrt(denom2)
  except ZeroDivisionError:
      return 0



dataframe = readJSONFile(fullFileSpec, sentimentCalculation)

dataframe.dropna(inplace = True)

if (sentimentCalculation == 'anew'):
    # here we start the emotionprediction model
    model = EmotionPredictor(classification="ekman", setting="mc", use_unison_model=True)

    # here we take from data frame from all the reviewText from our data frame with a loop,
    # and then we save it in a value we take the data from the first to last user reviewText
    value = dataframe.iloc[:]["reviewText"]

    # then we run the model and save the result to probabilities

    probabilities = model.predict_probabilities(value)

    # here we add to our existing df the new columns with ekmans emotions for every users reviewText
    dataframe['anger'] = probabilities.loc[:].Anger
    dataframe['disgust'] = probabilities.loc[:].Disgust
    dataframe['fear'] = probabilities.loc[:].Fear
    dataframe['joy'] = probabilities.loc[:].Joy
    dataframe['sadness'] = probabilities.loc[:].Sadness
    dataframe['surprise'] = probabilities.loc[:].Surprise

    # now we print the new df with our 3  columns we need in this case
    #  reviewerName, reviewText, overall plus all the feelings from above
    # for index, row in df.iterrows():
    #    print(index)
    #    print(row)

    # save the updated df to json file
    dataframe.to_json('User_rating_review.json', orient='split', compression = 'infer')
    dataframe.to_csv('User_rating_review.csv', sep='\t')
    #
    # print("original dataframe")
    # print(dataframe)


# from out dataframe we remove duplicate users
users = dataframe['reviewerID'].unique()
numUsers = len(users)
CS_similarities = [[0]*numUsers for i in range(numUsers)]
# for the prediction function
CS_similarities_sum = [[0]*numUsers for i in range(numUsers)]

reviews = []
for i in range(numUsers):
    reviews.append(dataframe[dataframe.reviewerID.eq(users[i])])
print("printing user information")
print("-- Skipped")
# print(reviews)
print("data frame for testing")
print("-- Skipped")
# for index, row in dataframe.iterrows():
#    print(index)
#    print(row)



"""
here we need 2 df's the one is for the original df_Sim with the original values 
the second one df_Sim_pearson, we use to to find mean of all values of the original df and the pass the into 
the function

"""


PEARSON_similarities = [[0]*numUsers for i in range(numUsers)]


# here we are finding the mean of all values to use the later in simPFweights function
# we need to compute averages for each of the dimensions
# pearson mean have the mean values of all users!!!!
PEARSON_means = []
for i in range(numUsers):
    PEARSON_means.append({})
    for dimension in weights.keys():
        sum = 0
        counter = 0
        for rindex in range(len(reviews[i].index)):
            sum = sum + reviews[i].iloc[rindex][dimension]
            counter = counter + 1
        PEARSON_means[i][dimension] = sum / counter

print("Pearson means:")
print(PEARSON_means);

# df_numeric = df_Sim.select_dtypes(exclude=['object'])
# df_numeric.drop(['asin'], axis=1)

# tmp value to store the pearson similarity
nn_pearson_similarities = 0

# Calculating only similarities for the first NUM_PREDICTIONS_TO_COMPUTE  users in the dataset

if (numUsers > NUM_PREDICTIONS_TO_COMPUTE):
    numSimilaritiestocompute = NUM_PREDICTIONS_TO_COMPUTE
else:
    numSimilaritiestocompute = numUsers

for i in range(numSimilaritiestocompute):
  for j in range(numUsers):
      if (i == j):
          PEARSON_similarities[i][j] = 1
      elif (i < j):
          # the list have the similarity between users[i][j]
          # new we are importing pearson similarity between users[i][j] to our dataframe to the new column "user[i][j]_similarities"
          nn_pearson_similarities = simPFweights(reviews[i], reviews[j], PEARSON_means[i], PEARSON_means[j], weights, scalings)
          # print(simpfweights(reviews_pearson[i], reviews_pearson_mean[k], reviews_pearson[j], reviews_pearson_mean[z], weights, scalings))
          # here we are saying the result of the function to df_sim_pearson
          PEARSON_similarities[i][j] = nn_pearson_similarities
          PEARSON_similarities[j][i] = nn_pearson_similarities

for i in range(numUsers):
  print(PEARSON_similarities[i])

# user_predct = input("witch user you want to find the prediction?")

print("=============================")
print("calculating cosine sim")
print("=============================")


for i in range(numSimilaritiestocompute):
    for j in range(numUsers):
        if (i == j):
            CS_similarities[i][j] = 1
        elif (i < j):
            # the list have the similarity between users[i][j]
            # now here we are calling the function cosine similarity to find the nearest_neighbours
            nn_cosine_similarities = simCFweights(reviews[i], reviews[j], weights, scalings)
            # new we are importing cosine similarity between users[i][j] to our dataframe to
            # the new column "user[i][j]_similarities"
            CS_similarities[i][j] = nn_cosine_similarities
            CS_similarities[j][i] = nn_cosine_similarities

print("CS similarities")
for i in range(numUsers):
        print(CS_similarities[i])

print("=============================")
print("calculating the pearson sim")
print("=============================")

# PREDICTION for cosine similarity

CS_SIMILARITY_THRESHOLD = 0.2
while (CS_SIMILARITY_THRESHOLD <= 0.8):
  total_computed = 0
  total_abs_error = 0
  random.seed()


  print("=============================")
  print("calculating cosine predictions, CS_SIMILARITY_THRESHOLD = " + str(CS_SIMILARITY_THRESHOLD))
  print("=============================")


  for i in range(NUM_PREDICTIONS_TO_COMPUTE):
    user = random.randint(0, numSimilaritiestocompute - 1)
    reviewNo = random.randint(0, len(reviews[user]) - 1)
    item = reviews[user].iloc[reviewNo].asin
    actualRating = reviews[user].iloc[reviewNo].overall
  
    # Compute cosine prediction
    numerator = 0
    denominator = 0
    for V in range(numUsers):
      if (V == user) or (CS_similarities[user][V] < CS_SIMILARITY_THRESHOLD):
        continue
      Vreview = reviews[V].loc[reviews[V]['asin'] == item]
      if (len(Vreview) == 0):
        continue
      numerator = numerator + CS_similarities[user][V] * (Vreview.iloc[0]['overall'] - PEARSON_means[V]['overall'])
      denominator = denominator + CS_similarities[user][V]
      
    if (denominator > 0):
      total_computed = total_computed + 1
      prediction = PEARSON_means[user]['overall'] + numerator / denominator
      error = prediction - actualRating
      total_abs_error = total_abs_error + abs(error)

  print("Computed " + str(total_computed) + " predictions for cosine similarity, CS_SIMILARITY_THRESHOLD = " + str(CS_SIMILARITY_THRESHOLD) + "mean absolute error = " + str(total_abs_error / total_computed))
  CS_SIMILARITY_THRESHOLD = CS_SIMILARITY_THRESHOLD + 0.1


# PREDICTION for pearson similarity
PEARSON_SIMILARITY_THRESHOLD = 0.0
while (PEARSON_SIMILARITY_THRESHOLD <= 0.7):
  print("=============================")
  print("calculating pearson predictions, PEARSON_SIMILARITY_THRESHOLD = " + str(PEARSON_SIMILARITY_THRESHOLD))
  print("=============================")


  total_computed_pearson = 0
  total_abs_error_pearson = 0
  random.seed()

  NUM_PREDICTIONS_TO_COMPUTE_PEARSON = 500

  for i in range(NUM_PREDICTIONS_TO_COMPUTE_PEARSON):
    user = random.randint(0, numSimilaritiestocompute - 1)
    reviewNo = random.randint(0, len(reviews[user]) - 1)
    item = reviews[user].iloc[reviewNo].asin
    actualRating = reviews[user].iloc[reviewNo].overall

    # Compute pearson prediction
    numerator = 0
    denominator = 0
    for V in range(numUsers):
        if (V == user) or (PEARSON_similarities[user][V] < PEARSON_SIMILARITY_THRESHOLD):
            continue
        Vreview = reviews[V].loc[reviews[V]['asin'] == item]
        if (len(Vreview) == 0):
            continue
        numerator = numerator + PEARSON_similarities[user][V] * (Vreview.iloc[0]['overall'] - PEARSON_means[V]['overall'])
        denominator = denominator + PEARSON_similarities[user][V]

    if (denominator > 0):
        total_computed_pearson = total_computed_pearson + 1
        prediction = PEARSON_means[user]['overall'] + numerator / denominator
        error = prediction - actualRating
        total_abs_error_pearson = total_abs_error_pearson + abs(error)

  try:
      print("Computed " + str(total_computed_pearson) + " predictions for pearson similarity PEARSON_SIMILARITY_THRESHOLD = " + str(PEARSON_SIMILARITY_THRESHOLD) + ", mean absolute error = " + str(total_abs_error_pearson / total_computed_pearson))

  except ZeroDivisionError:
      print("Computed " + str(total_computed_pearson) + " predictions for pearson similarity, mean absolute error = " + str(0))

  PEARSON_SIMILARITY_THRESHOLD = PEARSON_SIMILARITY_THRESHOLD + 0.1
