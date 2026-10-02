# Changes log from version find_similar
# A.
# The threshold for Pearson and Cosine Similarity changed to 0.6 to 1, so as to only consider positive similarities
#
# B.
# Computes PURE (original) similarity without taking into account the emotions derived from the review text.
#
# C.
# Gets rid of the random selection of users. The first X are selected according to terminal's arguments passed
#
# D.
# This version introduces PARALLELISM
#
# E.
# This version adds the ability to enter the number of users to handle through a command line argument.
# If ALL argument is entered, then the estimation involves all users.
# e.g. C:\python .\find_similar.py 100 1
# find similarities and predictions just for the first 100 users of dataset 1
# e.g. C:\python .\find_similar.py ALL 1
# find similarities and predictions just for ALL users of dataset 1
#
# ATTENTION. The program has two options, either there are TWO arguments or NONE. NO attempt to catch wrong use of the program.

# configuration
# -------------

# Here follows the Directories where the Data resides
#
# DIRECTORY = r'C:/Users/DS-pc/PycharmProjects/amazon_recomendations/twitter-emotion-recognition-master/'
# DIRECTORY = r'C:/Users/laptop/Desktop/amazon_recomendations/twitter-emotion-recognition-master/'
# DIRECTORY = r'/home/local/costas/classes/ptyx/michailidis/2021-07-23/amazon_recomendations_2.3.2/'
DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Exports/'
DIRECTORY_RESULT = r'C:/PHD/Paper_michailidis/Results/'

#  The file in DIRECTORY_SRC that stores the data under test
#
defaultFile_src = 'Musical_Instruments_5_ratings.json'
#  defaultFile = 'Magazine_Subscriptions.json'
#  defaultFile = 'Video_Games_5.json'

# The file in DIRECTORY_RESULT that stores the data with full predictions
#
defaultFile_result = defaultFile_src[:-13]
# defaultFile_src = 'Magazine_Subscriptions.json'
# defaultFile_src = 'Video_Games_5.json'

#  Choosing the type file to read (anew=create new file from scratch, cashed= open an already created file
defaultSentimentCalculation = 'cashed'

# -------------
# end of configuration


# Imports
import math
import os
# import tensorflow as tf     #unused! To remove?
os.environ['KERAS_BACKEND'] = 'theano'
os.environ["THEANO_FLAGS"] = "linker=py"    # new be Dennis-Chatgpt
import pandas as pd
import gzip
import numpy as np          # unused! To remove?
# from keras import backend as K  # unused! To remove?
# from parse import parse
import time

# emotion predictor lib must have it locally
# emotion predictor path in drive the path must be inside the folder

# from emotion_predictor import EmotionPredictor
# from pandas.io.json import json_normalize #unused! To remove?
import json
import sys
# import random

# from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

# end of imports

#  Store at df columns: reviewerName','reviewerID','reviewText','overall',asin (the code of the object rated), emotions
#  Keep original function, differs from the one at data_management.py
#  NOW we use 'cashed', as data has been already created.


def readJSONFile(filespec, filetype):
    # if file type = 'anew', we use this code
    if (filetype == 'anew'):
        df0 = pd.read_json(filespec, lines=True)
        df = pd.DataFrame(df0, columns=['reviewerName', 'reviewerID', 'reviewText', 'overall', 'asin'])
    else:
        # else, if file type = cached, use this code
        with open(filespec)as json_data:
            data = json.load(json_data)
        df = pd.DataFrame(data['data'])
        df.columns = data['columns']

    return df


#  DON'T LOOK
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


#  DON'T LOOK
def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


# Similarities. Estimate how similar two users are, in two ways: a)Pearson similarity b) Cosine Similarity
#
#
#                           simPFweights
#
# in this function we calculate the Pearson similarity of all pair of the dataframe
# in reviews1_mean,reviews2_mean we have the mean values of the rating
#
# reviews1, reviews2: Two Arrays of all reviews of reviewer1 and reviewer2. We create the product of the these products
#                     so as to find all review combinations of the two reviewers.
# reviews1_mean, reviews2_mean: Mean score review of reviewer1 and reviewer2. There is a mean for each emotion and an overall one (they total to seven means)
# weights: The weight assigned to each emotion, plus overall (they total to seven)
# scaling: Scaling of absolute values (as overall rating has a different scale [0..5] ranking that emotions [0..1]

def simPFweights(reviews1, reviews2, reviews1_mean, reviews2_mean, weights, scalings):
    numCommon = 0   # Number of common items
    nominator = 0   # Deviation from mean value (*weights *scaling)
    denom1 = 0      #
    denom2 = 0      #

    # for each user check if the item he rated, has also been rated by other users
    for index, row in reviews1.iterrows():  # Index: Index on reviewer1 ratings. Row: The elements of each row

        otherUserReview = reviews2[reviews2.asin.eq(row.asin)]  # find users rated asin (item of reviewer1)

        # check if the same item has been rated by user2
        if (len(otherUserReview) > 0):
            # same product rated
            numCommon = numCommon + 1           # find number of common items, between the two reviewers

            # implementation of the function according to sim-metrics
            for dimension in weights:
                nominator = nominator + (((row[dimension] - reviews1_mean[dimension]) * (otherUserReview.iloc[0][dimension] - reviews2_mean[dimension]))) * weights[dimension] * scalings[dimension]
                denom1 = denom1 + ((row[dimension] - reviews1_mean[dimension]) ** 2) * weights[dimension] * scalings[dimension]
                denom2 = denom2 + ((otherUserReview.iloc[0][dimension] - reviews2_mean[dimension]) ** 2) * weights[dimension] * scalings[dimension]
                # print("PEARSON:", row[dimension], " ", reviews1_mean[dimension], " ", otherUserReview.iloc[0][dimension], " ", reviews2_mean[dimension], " ", weights[dimension], " ", scalings[dimension], " ", denom1, " ", denom2)
                # input("Press Enter to continue...")

    # print("DENNIS 5,  Nominator:", nominator, " Denominator1:", denom1, " Denominator2:", denom2)
    # try:
    #     # print("Denominator CORRECT: ", denom1, " ", denom2)
    #     return nominator / (math.sqrt(denom1 * denom2))
    # except ZeroDivisionError:
    #     # print("Denominator ERROR: ", denom1, " ", denom2)
    #     return 0
    denom_product = denom1 * denom2
    if denom_product == 0:
        return 0
    return nominator / math.sqrt(denom_product)

# Similarities. Estimate how similar two users are, in two ways: a)Pearson similarity b) Cosine Similarity
#
#
#                           simCFweights
#
# in this function we calculate the cosine similarity of all pair of the dataframe


def simCFweights(reviews1, reviews2, weights, scalings):
    numCommon = 0
    nominator = 0
    denom1 = 0
    denom2 = 0

    # for each user check if the item he rated, has also been rated by other users
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
    # Kyriakos
    # try:
    #     return nominator / math.sqrt(denom1) / math.sqrt(denom2)
    # except ZeroDivisionError:
    #     return 0

    # Mine
    if denom1 == 0 or denom2 == 0:
        return 0
    return nominator / math.sqrt(denom1) / math.sqrt(denom2)


def Similarities(args):

    global USER_mean_rating

    start, end = args

    print(f"Start= {start}, End= {end}")

    """
    #
    #   PEARSON SIMILARITY
    #
    # here we need 2 df's the one is for the original df_Sim with the original values
    # the second one df_Sim_pearson, we use to find mean of all values of the original df and the pass the into
    # the function
    #
    """

    # df_numeric = df_Sim.select_dtypes(exclude=['object'])
    # df_numeric.drop(['asin'], axis=1)

    start_time_e = time.perf_counter()

    # tmp value to store the pearson similarity
    # nn_pearson_similarities = 0

    for i in range(start, end):
        # start_time_e1 = time.perf_counter()
        for j in range(numUsers):
            if (i == j):
                PEARSON_similarities[i][j] = 1
            # elif (i < j):
            else:
                # the list have the similarity between users[i][j]
                # new we are importing pearson similarity between users[i][j] to our dataframe to the new
                # column "user[i][j]_similarities"
                nn_pearson_similarities = simPFweights(reviews[i], reviews[j], USER_mean_rating[i], USER_mean_rating[j], weights, scalings)
                # print(simpfweights(reviews_pearson[i], reviews_pearson_mean[k], reviews_pearson[j], reviews_pearson_mean[z], weights, scalings))
                # here we are saying the result of the function to df_sim_pearson
                PEARSON_similarities[i][j] = nn_pearson_similarities
                # PEARSON_similarities[j][i] = nn_pearson_similarities
                # print("i= ", i, " j= ", j)

        # end_time_e1 = time.perf_counter()
        # execution_time_e1 = end_time_e1 - start_time_e1
        # print(f"E1 Execution Time: {execution_time_e1} seconds")
    # print(PEARSON_similarities)
    end_time_e = time.perf_counter()
    execution_time_e = end_time_e - start_time_e
    print(f"E Execution Time: {execution_time_e} seconds")

    """
    #
    #   COSINE SIMILARITY
    #   ESTIMATE SIMILARITIES AMONG ALL REVIEWERS
    #
    """

    start_time_f = time.perf_counter()

    for i in range(start, end):
        # start_time_f1 = time.perf_counter()
        for j in range(numUsers):
            if (i == j):
                CS_similarities[i][j] = 1
            # elif (i < j):
            else:
                # the list have the similarity between users[i][j]
                # now here we are calling the function cosine similarity to find the nearest_neighbours
                nn_cosine_similarities = simCFweights(reviews[i], reviews[j], weights, scalings)
                # new we are importing cosine similarity between users[i][j] to our dataframe to
                # the new column "user[i][j]_similarities"
                CS_similarities[i][j] = nn_cosine_similarities
                # CS_similarities[j][i] = nn_cosine_similarities
        # end_time_f1 = time.perf_counter()
        # execution_time_f1 = end_time_f1 - start_time_f1
        # print(f"F1 Execution Time: {execution_time_f1} seconds")

    end_time_f = time.perf_counter()
    execution_time_f = end_time_f - start_time_f
    print(f"F Execution Time: {execution_time_f} seconds")

    # print("CS similarities")
    # for i in range(numSimilaritiestocompute):
    #     print(CS_similarities[i])

    return start, end, PEARSON_similarities[start:end], CS_similarities[start:end]


def Simulation(args):

    global USER_mean_rating

    start, end, threshold = args

    print(f"Start= {start}, End= {end}")

    """
    #
    #   PREDICTION for PEARSON similarity
    #   ESTIMATE SIMILARITIES AMONG ALL REVIEWERS
    #
    """
    start_time_h = time.perf_counter()

    PEARSON_SIMILARITY_THRESHOLD = threshold
    # while (PEARSON_SIMILARITY_THRESHOLD <= 1):
    print("=============================")
    print("calculating pearson predictions, PEARSON_SIMILARITY_THRESHOLD = " + str(PEARSON_SIMILARITY_THRESHOLD))
    print("=============================")

    total_computed_pearson = 0
    total_abs_error_pearson = 0

    # for i in range(numUsers):
    for i in range(start, end):
       # start_time_h1 = time.perf_counter()

       #
       # DENNIS:
       # INSTEAD of random USE THE ACTUAL i user. This ensures that the experiments can be validated repeatedly, meaning
       # same terminal input, produces same output.
       # The same holds for reviewNo. We take the LAST review of the user.
       # user = random.randint(0, numSimilaritiestocompute - 1)
       # reviewNo = random.randint(0, len(reviews[user]) - 1)

        user = i
        reviewNo = len(reviews[user]) - 1
        item = reviews[user].iloc[reviewNo].asin
        actualRating = reviews[user].iloc[reviewNo].overall
        # print(f"DENNIS 1 {reviewNo} {item} {actualRating} {numUsers}")
        # Compute pearson prediction
        numerator = 0
        denominator = 0
        # for V in range(start, end):
        for V in range(numUsers):
            if (V == user) or (PEARSON_similarities[user][V] < PEARSON_SIMILARITY_THRESHOLD):
                continue
            Vreview = reviews[V].loc[reviews[V]['asin'] == item]
            if (len(Vreview) == 0):
                continue
            # print(f"DENNIS 2, User {user}, V {V}, Similarity {PEARSON_similarities[user][V]}, Rate {Vreview.iloc[0]['overall']}, Mean {USER_mean_rating[V]['overall']} ")
            numerator = numerator + PEARSON_similarities[user][V] * (Vreview.iloc[0]['overall'] - USER_mean_rating[V]['overall'])
            denominator = denominator + PEARSON_similarities[user][V]
        # print(f"DENNIS 3 NUM/DENOM {numerator} {denominator}  USER {user}")
        if (denominator > 0):
            total_computed_pearson = total_computed_pearson + 1
            prediction = USER_mean_rating[user]['overall'] + numerator / denominator
            error = prediction - actualRating
            total_abs_error_pearson = total_abs_error_pearson + abs(error)
            # print("DENNIS 4, Pearson", user)
        # end_time_h1 = time.perf_counter()
        # execution_time_h1 = end_time_h1 - start_time_h1
        # print(f"H1 Execution Time: {execution_time_h1} seconds")

    try:
        print("Computed " + str(total_computed_pearson) + " predictions for pearson similarity PEARSON_SIMILARITY_THRESHOLD = " + str(PEARSON_SIMILARITY_THRESHOLD) + ", mean absolute error = " + str(total_abs_error_pearson / total_computed_pearson))

    except ZeroDivisionError:
        print("Computed " + str(total_computed_pearson) + " predictions for pearson similarity, mean absolute error = " + str(0))

    end_time_h = time.perf_counter()
    execution_time_h = end_time_h - start_time_h
    print(f"H Execution Time: {execution_time_h} seconds")

    # for i in range(numSimilaritiestocompute):
    #   print(PEARSON_similarities[i])
    # user_predct = input("which user you want to find the prediction?")

    """
    #
    #   PREDICTION for COSINE similarity
    #   ESTIMATE SIMILARITIES AMONG ALL REVIEWERS
    #
    """

    print("=============================")
    print("calculating the cosine sim")
    print("=============================")

    # PREDICTION for cosine similarity

    start_time_g = time.perf_counter()

    CS_SIMILARITY_THRESHOLD = threshold
    # while (CS_SIMILARITY_THRESHOLD <= 1):
    total_computed = 0
    total_abs_error = 0
    # random.seed()

    print("=============================")
    print("calculating cosine predictions, CS_SIMILARITY_THRESHOLD = " + str(CS_SIMILARITY_THRESHOLD))
    print("=============================")

    # for i in range(numUsers):
    for i in range(start, end):

        # start_time_g1 = time.perf_counter()

        #
        # DENNIS:
        # INSTEAD of random USE THE ACTUAL i user. This ensures that the experiments can be validated repeatedly, meaning
        # same terminal input, produces same output.
        # The same holds for reviewNo. We take the LAST review of the user.
        # user = random.randint(0, numSimilaritiestocompute - 1)
        # reviewNo = random.randint(0, len(reviews[user]) - 1)

        user = i
        reviewNo = len(reviews[user]) - 1
        item = reviews[user].iloc[reviewNo].asin
        actualRating = reviews[user].iloc[reviewNo].overall
        # print(f"DENNIS 6 {reviewNo} {item} {actualRating} {numUsers}")

        # Compute cosine prediction
        numerator = 0
        denominator = 0
        # for V in range(start, end):
        for V in range(numUsers):
          # if ((V == 1) and (user == 14)) or ((V == 9) and (user == 17)) or ((V == 10) and (user == 27)):
          #    print("DENNIS 7c, User: ", user, " V: ", V, " CS_similarities[user][V]: ", CS_similarities[user][V])
          if (V == user) or (CS_similarities[user][V] < CS_SIMILARITY_THRESHOLD):
              # if ((V == 1) and (user == 14)) or ((V == 9) and (user == 17)) or ((V == 10) and (user == 27)):
              #     print("DENNIS 7a, User: ", user, " V: ", V, " CS_similarities[user][V]: ", CS_similarities[user][V])
              continue
          Vreview = reviews[V].loc[reviews[V]['asin'] == item]
          if (len(Vreview) == 0):
              # if ((V == 1) and (user == 14)) or ((V == 9) and (user == 17)) or ((V == 10) and (user == 27)):
              #     print("DENNIS 7b, User: ", user, " V: ", V, " len(Vreview: ", len(Vreview))
              continue
          # print(f"DENNIS 7, User {user}, V {V}, Similarity {CS_similarities[user][V]}, Rate {Vreview.iloc[0]['overall']}, Mean {USER_mean_rating[V]['overall']} ")
          numerator = numerator + CS_similarities[user][V] * (Vreview.iloc[0]['overall'] - USER_mean_rating[V]['overall'])
          denominator = denominator + CS_similarities[user][V]
        # print(f"DENNIS 8 NUM/DENOM {numerator} {denominator}  USER {user}")
        if (denominator > 0):
          total_computed = total_computed + 1
          prediction = USER_mean_rating[user]['overall'] + numerator / denominator
          error = prediction - actualRating
          total_abs_error = total_abs_error + abs(error)
          # print("DENNIS 9, Cosine", user)
        # end_time_g1 = time.perf_counter()
        #
        # execution_time_g1 = end_time_g1 - start_time_g1
        # print(f"G1 Execution Time: {execution_time_g1} seconds")
    try:
        print("Computed " + str(total_computed) + " predictions for cosine similarity, CS_SIMILARITY_THRESHOLD = " + str(
                CS_SIMILARITY_THRESHOLD) + " mean absolute error = " + str(total_abs_error / total_computed))
    except ZeroDivisionError:
        print("Computed " + str(total_computed) + " predictions for cosine similarity, CS_SIMILARITY_THRESHOLD = " + str(
                CS_SIMILARITY_THRESHOLD) + " mean absolute error = " + str(0))

    # CS_SIMILARITY_THRESHOLD = CS_SIMILARITY_THRESHOLD + 0.1

    end_time_g = time.perf_counter()
    execution_time_g = end_time_g - start_time_g
    print(f"G Execution Time: {execution_time_g} seconds")

    return PEARSON_SIMILARITY_THRESHOLD, total_computed_pearson, total_abs_error_pearson, CS_SIMILARITY_THRESHOLD, total_computed, total_abs_error


"""
# START OF MAIN PROGRAM
"""

# sys.stdout = open(DIRECTORY_RESULT + defaultFile_result + "_results_pure_denis_parallel.txt", "w")
start_time_a0 = time.perf_counter()

# Set DIRECTORY AS default directory

os.chdir(DIRECTORY_SRC)

# A. weights help us to change the impact of a specific emotion or the overall rating for our simulation.
# For example for some emotions that we can assume they have bigger impact on out simulation
# then the same stands for the overall rating
#
# B. we need the scaling because our emotion function have fields from [0,1] with sum of 1 that means
# it's not possible to implement it on the amazon dataset (like most datasets) because the range is [1 ,5]
# that is why we need the scaling, on different datasets it must change
# ASK KYRIAKO OR READ PAPER

weights = {'overall': 0.5, 'anger': 0.6, 'disgust': 0.3, 'fear': 0.1, 'joy': 0.6, 'sadness': 0.4, 'surprise': 0.2}  # emotion version
scalings ={'overall': 0.1, 'anger': 1, 'disgust': 1, 'fear': 1, 'joy': 1, 'sadness': 1, 'surprise': 1}              # emotion version

if len(sys.argv) >= 4:
    WeightScale=sys.argv[3]
    if WeightScale.upper()=='P':
        weights = {'overall': 1, 'anger': 0.0, 'disgust': 0.0, 'fear': 0.0, 'joy': 0.0, 'sadness': 0.0, 'surprise': 0.0}    # pure version
        scalings = {'overall': 1, 'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0}               # pure version

# Choose whether you enter the data file name as an argument at terminal or by the configuration part
# of the program

if len(sys.argv) >= 3:
    file_code = sys.argv[2]
    if file_code == '1':
        file = 'Musical_Instruments_5_ratings.json'
    elif file_code == '2':
        file = 'Magazine_Subscriptions_ratings.json'
    elif file_code == '3':
        file = 'Video_Games_5_ratings.json'
    elif file_code == '4':
        file = 'Amazon_Instant_Video_5_ratings.json'
    elif file_code == '5':
        file = 'Digital_Music_5_ratings.json'
    elif file_code == '6':
        file = 'Automotive_5_ratings.json'
    else:
        file = 'Musical_Instruments_5_ratings.json'
else:
    file = defaultFile_src

# Define the full path name of the file the stores the data

fullFileSpec = DIRECTORY_SRC + file

# Choose whether you enter the type of Sentiment as an argument at terminal or by the configuration part
# of the program

sentimentCalculation = defaultSentimentCalculation

# Function to read data from JSON file. Returns a data frame
# read cached JSON file

end_time_a0 = time.perf_counter()
execution_time_a0 = end_time_a0 - start_time_a0
print(f"A0 Execution Time: {execution_time_a0} seconds")


# Read existing file with the ranking of the users

start_time_a=time.perf_counter()
dataframe = readJSONFile(fullFileSpec, sentimentCalculation)

# from out dataframe we find unique users
users = dataframe['reviewerID'].unique()
items = dataframe['asin'].unique()
numUsers = len(users)
numItems = len(items)
print(f"Number of users: {numUsers}")
print(f"Number of items: {numItems}")
end_time_a = time.perf_counter()
execution_time_a = end_time_a - start_time_a
print(f"A Execution Time: {execution_time_a} seconds")

start_time_c = time.perf_counter()

# Create an array (reviews) of arrays (for each user, its ratings)

reviews = []
for i in range(numUsers):
    reviews.append(dataframe[dataframe.reviewerID.eq(users[i])])
    # print(i, len(reviews[i]))

# print("printing user information")
# print("-- Skipped")
# # print(reviews)
# print("data frame for testing")
# print("-- Skipped")
# for index, row in dataframe.iterrows():
#    print(index)
#    print(row)

end_time_c = time.perf_counter()
execution_time_c = end_time_c - start_time_c
print(f"C Execution Time: {execution_time_c} seconds")


# How many prediction to compute from the whole set of possible predictions
# Calculating only similarities for the first numSimilaritiestocompute  users in the dataset

if len(sys.argv) >= 3:
    if sys.argv[1]=="ALL":
        numSimilaritiestocompute = numUsers
    else:
        numSimilaritiestocompute = int(sys.argv[1])
else:
    numSimilaritiestocompute = numUsers

start_time_d = time.perf_counter()

# here we find the mean of all values to use the later in simPFweights function
# we need to compute averages for each of the dimensions (emotions+overall)
# pearson means have the mean values of all users!!!!

USER_mean_rating = []
for i in range(numUsers):
    # start_time_d1 = time.perf_counter()
    USER_mean_rating.append({})
    for dimension in weights.keys():
        sum = 0
        counter = 0
        for rindex in range(len(reviews[i].index)):
            sum = sum + reviews[i].iloc[rindex][dimension]
            counter = counter + 1
        USER_mean_rating[i][dimension] = sum / counter

# print("Pearson means:")
# print(USER_mean_rating)

# --- ADD THESE THREE LINES ---
# ---  ERASE dataframe that is no longer needed
#
import gc               # SECOND CORRECTION
del dataframe
gc.collect()
# -----------------------------

end_time_d = time.perf_counter()
execution_time_d = end_time_d - start_time_d
print(f"D Execution Time: {execution_time_d} seconds")

# Define threads for parallel execution
Threads_Num=12     # Threads_Num: Number of threads
chunk_size=numSimilaritiestocompute // Threads_Num  # chunk_size: The users assigned to each thread


def init_simulation_worker(pearson_similarities_data, cs_similarities_data):

    # Runs once, at startup, in every Simulation worker process (whether
    # spawned or forked). Sets `pearson_similarities` and `cs_similarities` as globals in
    # that worker so Simulation() can read them, without relying on the
    # worker having re-executed the (now-parallel) build step itself.

    global PEARSON_similarities, CS_similarities

    PEARSON_similarities=pearson_similarities_data
    CS_similarities = cs_similarities_data


if __name__ == "__main__":

    start_time_b1 = time.perf_counter()
    # Initialize matrices for the prediction function.

    start_time_ba = time.perf_counter()

    # for the prediction function
    # PEARSON_similarities = [[0] * numUsers for i in range(numUsers)]
    # CS_similarities = [[0] * numUsers for i in range(numUsers)]

    PEARSON_similarities = np.zeros((numUsers, numUsers), dtype=np.float64) # SECOND CORRECTION
    CS_similarities = np.zeros((numUsers, numUsers), dtype=np.float64)

    Similarity_chunks = []
    for i in range(Threads_Num):
        start = i * chunk_size
        end = start + chunk_size
        if i == Threads_Num - 1:
            end = numSimilaritiestocompute
        print(f"DENNIS 11 Chunk ID: {i}, start: {start}, end: {end}")
        Similarity_chunks.append((start, end))
        # print(f"Start= {start}, End= {end} ")
    # print(f"Den is: {den}")
    # print(f"Chunk size is: {chunk_size}")
    with ProcessPoolExecutor(max_workers=Threads_Num, initializer=init_simulation_worker, initargs=(PEARSON_similarities, CS_similarities)) as executor:
        Similarities_results=list(executor.map(Similarities, Similarity_chunks))

    # Combine the results returned by all workers
    for start, end, pearson_rows, cosine_rows in Similarities_results:
        PEARSON_similarities[start:end] = pearson_rows
        CS_similarities[start:end] = cosine_rows

    end_time_ba= time.perf_counter()
    execution_time_ba= end_time_ba - start_time_ba
    print(f"Ba Execution Time: {execution_time_ba} seconds")

    SIMILARITY_THRESHOLD=0.9
    while (SIMILARITY_THRESHOLD <= 0.9):

        start_time_b0 = time.perf_counter()
        chunks = []

        for i in range(Threads_Num):
            start = i * chunk_size
            end = start + chunk_size
            if i == Threads_Num - 1:
                end = numSimilaritiestocompute
            # print(f"Chunk ID: {i}, start: {start}, end: {end}, similarity: {SIMILARITY_THRESHOLD}")
            chunks.append((start, end, SIMILARITY_THRESHOLD))
            # print(f"Start= {start}, End= {end} ")
        # print(f"Den is: {den}")
        # print(f"Chunk size is: {chunk_size}")

        with ProcessPoolExecutor(max_workers=Threads_Num, initializer=init_simulation_worker, initargs=(PEARSON_similarities, CS_similarities)) as executor:
            Similarity_results=list(executor.map(Simulation, chunks))

        end_time_b0 = time.perf_counter()
        execution_time_b0 = end_time_b0 - start_time_b0
        print(f"B0 Execution Time: {execution_time_b0} seconds")
        print(f"Dennis 10, {Similarity_results}")
        all_results = {}  # results per threshold
        all_sums = {}     # sum per column, per threshold

        results_array = np.array(Similarity_results)    # shape: (N, 3). results_array, same as result but in an array form
        column_sums = np.sum(results_array, axis=0)     # shape: (3,).   holds the sum of every column
        # print(f"Dennis 11, {results}")
        # Store results to new arrays, per SIMILARITY_THRESHOLD
        all_results[SIMILARITY_THRESHOLD] = results_array
        all_sums[SIMILARITY_THRESHOLD] = column_sums

        print(f"Threshold {SIMILARITY_THRESHOLD} -> Sums: {column_sums}")

        SIMILARITY_THRESHOLD=round(SIMILARITY_THRESHOLD+0.1, 2)

        print("THE END")

    end_time_b1 = time.perf_counter()
    execution_time_b1 = end_time_b1 - start_time_b1
    print(f"B1 Execution Time: {execution_time_b1} seconds")
