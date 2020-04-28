import pandas as pd
import numpy as np
from pathlib import Path


#load dataset
data_path = Path("./data/movieLens/")
tags = pd.read_csv(data_path/"tags.csv")
links = pd.read_csv(data_path/"links.csv")
movies = pd.read_csv(data_path/"movies.csv")
ratings = pd.read_csv(data_path/"ratings.csv")
genome_tags = pd.read_csv(data_path/"genome-tags.csv")
genome_scores = pd.read_csv(data_path/"genome-scores.csv")

print(tags.head())