import pandas as pd
import numpy as np
from pathlib import Path
from surprise import Dataset

#load dataset
data_path = Path("/Users/jing/Desktop/ml-latest-small")
tags = pd.read_csv(data_path/"tags.csv") #userId,movieId,tag,timestamp
links = pd.read_csv(data_path/"links.csv") #movieId,imdbId,tmdbId
movies = pd.read_csv(data_path/"movies.csv") #movieId,title,genres
ratings = pd.read_csv(data_path/"ratings.csv")#userId,movieId,rating,timestamp
#genome_tags = pd.read_csv(data_path/"genome-tags.csv") #tagId,tag
#genome_scores = pd.read_csv(data_path/"genome-scores.csv") #movieId,tagId,relevance

# data = Dataset.load_builtin('ml-100k')
# print(data)
#print(tags.head())

#print(ratings.head())

#print(movies.head())
#inner join

# ratings_dict = {'itemID': list(ratings.movieId),
#                 'userID': list(ratings.userId),
#                 'rating': list(ratings.rating)}
# df = pd.DataFrame(ratings_dict)
#ratings = ratings[:200000]
df=ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

print(df.head())
