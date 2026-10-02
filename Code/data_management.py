
# configuration
# -------------

# Here follows the Directories where the Data resides
#
# DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Musical_Instruments_5/'
# DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Magazine_Subscriptions/'
# DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Video_Games_5/'
# DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Amazon_Instant_Video_5/'
DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Digital_Music_5/'

# The file in DIRECTORY_SRC that stores the data under test
#
# defaultFile_src = 'Musical_Instruments_5.json'
# defaultFile_src = 'Magazine_Subscriptions.json'
# defaultFile_src = 'Video_Games_5.json'
# defaultFile_src = 'Amazon_Instant_Video_5.json'
defaultFile_src = 'Digital_Music_5.json'

# Here follows the Directories where the Results reside
#

DIRECTORY_EXPORT = r'C:/PHD/Paper_michailidis/Exports/'


# The file in DIRECTORY_RESULT that stores the data with full predictions
#
# defaultFile_src = 'Magazine_Subscriptions.json'
defaultFile_export = defaultFile_src[:-5]
# defaultFile_src = 'Video_Games_5.json'

#  Choosing the Sentiment Calculation to perform
defaultSentimentCalculation = 'anew'


# -------------
# end of configuration

# Imports
# ---------

import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ["THEANO_FLAGS"] = "linker=py"   # new by Dennis-Chatgpt

import pandas as pd

# import numpy as np          #unused! To remove?

# emotion predictor lib must have it locally
# emotion predictor path in drive the path must be inside the folder

from emotion_predictor import EmotionPredictor

# import json
import sys
import time


# end of imports

start_time_a0 = time.perf_counter()

# Set DIRECTORY_SRC AS default DIRECTORY_SRC

os.chdir(DIRECTORY_SRC)

# Choose whether you enter the data file name as an argument at terminal or by the configuration part
# of the program

if len(sys.argv) >= 2:
    file_code=sys.argv[1]
    if file_code=='1':
        DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Musical_Instruments_5/'
        file='Musical_Instruments_5.json'
    elif file_code=='2':
        DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Magazine_Subscriptions/'
        file = 'Magazine_Subscriptions.json'
    elif file_code=='3':
        DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Video_Games_5/'
        file = 'Video_Games_5.json'
    elif file_code=='4':
        DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Amazon_Instant_Video_5/'
        file = 'Amazon_Instant_Video_5.json'
    elif file_code=='5':
        DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Digital_Music_5/'
        file = 'Digital_Music_5.json'
    elif file_code=='6':
        DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Automotive_5/'
        file = 'Automotive_5.json'
    elif file_code=='7':
        DIRECTORY_SRC = r'C:/PHD/Paper_michailidis/Data/Office_Products_5/'
        file = 'Office_Products_5.json'
    else:
        file = 'Musical_Instruments_5.json'
else:
    file = defaultFile_src

# Define the full path name of the file the stores the data

fullFileSpec = DIRECTORY_SRC + file

# Function to read data from JSON file. Returns a data frame
# read cached JSON file

end_time_a0 = time.perf_counter()
execution_time_a0 = end_time_a0 - start_time_a0
print(f"A0 Execution Time: {execution_time_a0} seconds")

# Store at df columns: reviewerName','reviewerID','reviewText','overall',asin (the code of the object rated)


def readJSONFile(filespec):
    # File type = 'anew', at all times, so we use this code

    df0 = pd.read_json(filespec, lines=True)
    df = pd.DataFrame(df0, columns=['reviewerName', 'reviewerID', 'reviewText', 'overall', 'asin'])
    return df


start_time_a = time.perf_counter()

dataframe = readJSONFile(fullFileSpec)

end_time_a = time.perf_counter()

execution_time_a = end_time_a - start_time_a
print(f"A Execution Time: {execution_time_a} seconds")

start_time_b = time.perf_counter()
dataframe.dropna(inplace=True)            # remove rows having empty fields

# here we start the emotion prediction model
model = EmotionPredictor(classification="ekman", setting="mc", use_unison_model=True)

# here we take from data frame from all the reviewText from our data frame with a loop,
# and then we save it in a value we take the data from the first to last user reviewText
value = dataframe.iloc[:]["reviewText"]

# then we run the model and save the result to probabilities

probabilities = model.predict_probabilities(value)

# here we add to our existing df the new columns with ekmans emotions for every user reviewText
dataframe['anger'] = probabilities.loc[:].Anger
dataframe['disgust'] = probabilities.loc[:].Disgust
dataframe['fear'] = probabilities.loc[:].Fear
dataframe['joy'] = probabilities.loc[:].Joy
dataframe['sadness'] = probabilities.loc[:].Sadness
dataframe['surprise'] = probabilities.loc[:].Surprise

end_time_b = time.perf_counter()
execution_time_b = end_time_b - start_time_b
print(f"B Execution Time: {execution_time_b} seconds")


# Keep only users with at least 10 reviews

start_time_c = time.perf_counter()

MINIMUM_REVIEWS=10
print("Passed 01")
valid_users = dataframe['reviewerID'].value_counts()
print("Passed 02")
valid_users = valid_users[valid_users >= MINIMUM_REVIEWS].index
print("Passed 03")
df_filtered = dataframe[dataframe['reviewerID'].isin(valid_users)]
print("Passed 04")

end_time_c = time.perf_counter()
execution_time_c = end_time_c - start_time_c
print(f"C Execution Time: {execution_time_c} seconds")

start_time_d = time.perf_counter()

defaultFile_export = file[:-5]

# save the updated df to both json + cvs files. ALL USERS
dataframe.to_json(DIRECTORY_EXPORT + defaultFile_export+'_ratings.json', orient='split', compression='infer')
dataframe.to_csv(DIRECTORY_EXPORT + defaultFile_export+'_ratings.csv', sep='\t')

# save the updated df to both json + cvs files.  ONLY USER WITH MINIMUM_REVIEWS=10
# df_filtered.to_json(DIRECTORY_EXPORT + defaultFile_export+'_ratings.json', orient='split', compression='infer')
# df_filtered.to_csv(DIRECTORY_EXPORT + defaultFile_export+'_ratings.csv', sep='\t')

end_time_d = time.perf_counter()
execution_time_d = end_time_d - start_time_d
print(f"D Execution Time: {execution_time_d} seconds")
