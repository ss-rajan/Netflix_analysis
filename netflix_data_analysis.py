# -*- coding: utf-8 -*-
"""Netflix Data Analysis

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/siddharth164/netflix-data-analysis.bb6965bb-3b29-4b52-aef1-8834ea0de86b.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20250421/auto/storage/goog4_request%26X-Goog-Date%3D20250421T205813Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D511eebedd1788109ac376f978e8dadb430985e5d6f077cfa14cfdce6f9a89bf88e84ddea12eb140ad49ac97db3f7fe118ee32e4f8ad35df2f4d8fb5b1f1487e18869dd16e8ad6be5aa41d4fe821592094632cb1228952926f04925837d33f212bc74a11cbc007fbf59c6fe120f0c18105d34797f7728ef0b32602d8fa9058f97e21bcffa11aafa22690dc29deaa30eed158acf5d5001fe5b13fe16bce90eeecb1fa088dfab25718a1d374f46b90845cc88d8c6ba3b533495f66e6f435b7bfc2de8f8d9559a018bf9fd11f98f18a93334e903a5248f24bcf700a60545c623edc5237e2d3bbfe507326a5d2258fc5d67886a1dccd4bdac7abbd5002f3c738754e5
"""

import os
import numpy as np
import pandas as pd

#For data visualization
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")

"""# Introduction

In this project, I will explore and visualize a Netflix dataset containing information about various shows and movies available on the platform. The dataset includes key attributes such as title, director, cast, country of origin, release year, rating, duration, and genres. Using Seaborn, I will create insightful visualizations to uncover trends and patterns in Netflix's content library.

This project aims to analyze Netflix’s content library to uncover trends, patterns, and key insights. By visualizing data, we seek to understand Netflix’s content strategy, audience preferences, and market trends.
"""

#Loading the dataset
file = 'https://raw.githubusercontent.com/ss-rajan/Netflix_analysis/main/netflix_movies%20(1).csv'
df = pd.read_csv(file)

"""# Understanding the dataset"""

df.head() #Preview the datset

df.info() # Check column data types and null values

df.nunique() # Statistics for unique values

df.shape #Finding the rows and columns

# Finding range of the dataset

starting_range = df['release_year'].min()
ending_range = df['release_year'].max()
print(f"The dataset ranges from",starting_range, f"to", ending_range)

"""#  Data Analysis and Visualizations

The below section will have the following graphs developed by *seaborn*.
* Content Distribution
* Content per year
* Duration range of Movies
* Duration range of TV shows
* Contributions per country
* Movies by directior
* Top 20 longest movies
* Top 20 shortest movies
* Top 20 longest running TV shows
* Content per rating

## Content Distribution
"""

# Group by count of type
numtype = df.groupby(['type']).size().to_frame(name='count').reset_index()

#Calculate percentage
numtype['percentage'] = (numtype['count'] / numtype['count'].sum()) * 100
numtype

plt.title("Content Distribution")
sns.barplot(x="type", y="count",data = numtype)
for i, row in numtype.iterrows():
    plt.text(i, row["count"]+60, f"{row['percentage']:.1f}%", ha='center', fontsize=12, fontweight='bold')

plt.ylim(0, 8000)
plt.xlabel("Content Type", fontsize=12)
plt.ylabel("", fontsize=12)
plt.show()

"""## Content per year"""

# To find amount of content released per year

content_per_year = df['release_year'].value_counts().reset_index()

plt.figure(figsize=(10,8))
sns.lineplot(data=content_per_year, x='release_year', y='count', marker='o', color='blue')

plt.xlim(content_per_year['release_year'].min(), content_per_year['release_year'].max())  # Set x-axis limits

plt.title("Releases per year")
plt.xlabel("Release year")
plt.ylabel("Number of movies")
plt.show()

#To understand type of content released per year

per_year = df.groupby(['release_year', 'type']).size().to_frame(name='count')

per_year.tail(10).sort_values(by='release_year', ascending=False)

plt.title("Content per year")

sns.lineplot(data=per_year, x="release_year",y="count",hue='type')

plt.ylabel("Content")
plt.xlabel("Year")
plt.show()

"""## Duration range of movies"""

#To find duration of movies on Netflix dropping NaN values

movie_durations = df.loc[df['type'] == 'Movie']['duration'].str.replace(' min','').astype(float).reset_index(drop=True)

average_duration = movie_durations.mean()
print(f"Average movie duration: {average_duration:.2f} minutes")
movie_durations

plt.title("Distribution of Movie Durations (Minutes)")

sns.histplot(movie_durations, bins=40, kde=False, color="skyblue")

plt.ylabel("Number of movies")
plt.xlabel("Duration (Minutes)")
plt.show()

"""## Average duration of TV Shows"""

TV_dur = df[df["type"] == "TV Show"][["title", "duration"]].assign(duration=lambda x: x["duration"].str.replace(' Seasons| Season','',regex=True).astype(int))
TV_dur = TV_dur.sort_values(by="duration", ascending=False).reset_index(drop=True)
average_tvdur = TV_dur["duration"].mean()
print(f"Average TV Show season spans: {average_tvdur:.2f} seasons")
TV_dur

season_counts = (
    TV_dur["duration"]
    .value_counts()
    .sort_index()
    .rename_axis("Number of Seasons")
    .reset_index(name="Count of TV Shows")
    .set_index("Number of Seasons")
)

# Add percentage column
season_counts["Percentage"] = (season_counts["Count of TV Shows"] / season_counts["Count of TV Shows"].sum() * 100).round(1)

# Display the table
season_counts

season_counts.plot(kind='bar', y='Count of TV Shows', color="skyblue",legend=False)
plt.title("TV Shows by Number of Seasons")
plt.xlabel("Number of Seasons")
plt.ylabel("Count of TV Shows")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

"""## Contributions per country"""

# To analyse top 10 countries with most contributions

country_contri = df.country.value_counts().to_frame(name='Contributions')
country_contri = country_contri.iloc[:10]
country_contri

plt.title("Contributions per country")
sns.barplot(x=country_contri["Contributions"],y=country_contri.index)

plt.ylabel("Country")
plt.xlabel("Contributions")
plt.show()

"""## Movies by directior"""

# Module for text wrapping and formatting
import textwrap

mov_per_dir = df[df['type'] == 'Movie']['director'].value_counts().reset_index()
mov_per_dir = mov_per_dir.iloc[:10]

mov_per_dir

plt.figure(figsize=(10,6))
plt.title("Movies per director", fontsize =15)
sns.barplot (data = mov_per_dir, y = 'director', x ='count')
plt.xticks(ticks=range(0, int(mov_per_dir['count'].max()) + 3, 3))
for i, row in mov_per_dir.iterrows():
    plt.text(row["count"]+0.5,i, f"{row['count']}",ha='center', fontsize=12, fontweight='bold')

plt.xlabel("Count")
plt.ylabel("Director")
plt.show()

"""## Top 20 longest movies"""

movie_dur = df[df["type"] == "Movie"][["title", "duration"]].assign(duration=lambda x: x["duration"].str.replace(' min','').astype(float))
movie_dur_long = movie_dur.sort_values(by="duration", ascending=False).reset_index(drop=True)
top10_ml = movie_dur_long.head(10)
top10_ml

top10_ml['wrapped_titles'] = top10_ml['title'].apply(lambda x: "\n".join(textwrap.wrap(x, width=20)))  #wrapping text for better visual

plt.figure(figsize=(10, 8))
sns.barplot(y= "wrapped_titles", x= "duration", data = top10_ml)

# Customize the plot
plt.title("Longest movie durations", fontsize=14)
plt.ylabel('')
plt.xlabel("Duration (in mins)",fontsize=14)
plt.show()

"""## Top 20 shortest movies

"""

movie_dur_shrt = movie_dur.sort_values(by="duration").reset_index(drop=True)
top10_ms= movie_dur_shrt.head(10)

top10_ms['wrapped_titles'] = top10_ms['title'].apply(lambda x: "\n".join(textwrap.wrap(x, width=20)))  # 20 characters per line

plt.figure(figsize=(10,8))
sns.barplot(y= "wrapped_titles", x= "duration", data = top10_ms)

# Customize the plot
plt.title("Shortest movie durations", fontsize=14)
plt.ylabel('')
plt.xlabel("Duration (in mins)",fontsize=14)
plt.show()

"""## Top 20 longest running TV shows"""

top20_dur = TV_dur.head(20)
top20_dur

plt.figure(figsize=(8, 5))
sns.barplot(y= "title", x= "duration", data = top20_dur)

# Customize the plot
plt.title("Longest TV shows", fontsize=14)
plt.xlabel("Number of seasons")
plt.ylabel("Shows")
plt.show()

"""## Content per Rating"""

valid_ratings = [
    "G", "PG", "PG-13", "R", "NC-17", "NR", "UR",
    "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "TV-PG", "TV-14", "TV-MA"
]

# Filter only rows with valid ratings
rating_df = df[df["rating"].isin(valid_ratings)].copy()

# Merge closely related ratings
rating_df["rating"] = rating_df["rating"].replace({
    "PG": "TV-PG",         # Merge TV-PG and PG
    "TV-Y7-FV": "TV-Y7",   # Merge TV-Y7 and TV-Y7-FV
    "G": "TV-G",           # Merge TV-G and G
    "R": "TV-MA",          # Merge TV-MA and R
    "UR": "NR"             # Merge UR and NR
})

# Create maturity level groups
rating_df["group"] = rating_df["rating"].replace({
    "TV-MA": "Mature",
    "PG-13": "Mature",
    "NC-17": "Mature",
    "TV-14": "Teen",
    "TV-PG": "Teen",
    "TV-Y7": "Kids",
    "TV-Y": "Kids",
    "TV-G": "Kids",
    "NR": "NR"
})

# Count and calculate percentages
rating_counts = rating_df["group"].value_counts().reset_index()
rating_counts.columns = ["group", "count"]
rating_counts["percentages"] = (rating_counts["count"] / rating_counts["count"].sum()) * 100
rating_counts["percentages"] = rating_counts["percentages"].round(2)

rating_counts

plt.figure(figsize=(10, 6))
plt.title("Rating Distribution", fontsize=14, fontweight='bold')

sns.barplot(x="group", y="count", data=rating_counts, palette="pastel")

# Add count labels on top of bars
for i, row in rating_counts.iterrows():
    plt.text(i, row["count"] + 50, f"{row['count']}", ha='center', fontsize=12, fontweight='bold')

plt.ylim(0, rating_counts["count"].max() + 500)
plt.xlabel("Rating Group", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()

"""More info on parental guidelines can be found [here](https://rating-system.fandom.com/wiki/TV_Parental_Guidelines).

# Conclusions

For the given Netflix database, which spans from the year 1925 to 2021, there are a total of 8,807 titles consisting of both Movies and TV Shows. Supporting data such as Directors, Cast, Release Year, Rating, and more is also available for these titles.

Based on the data analysis and visualizations, the following insights were drawn:

* 69.6% of the total content are Movies, with the remaining 30.4% being TV Shows.
* A significant surge in the creation of content began in the early 2000s and continued to grow rapidly over the following two decades.
* The average movie duration is approximately 99.58 minutes, while 67% of TV shows consist of only one season.
* The United States is the largest contributor to the platform, followed by India, the United Kingdom, Japan, and South Korea.
* Indian director Rajiv Chilaka stands out as the most prolific contributor to the platform.
* Black Mirror: Bandersnatch holds the record for the longest movie duration due to its interactive format, while Grey's Anatomy is the longest-running TV show.
* About 51.12% of the content is intended for Mature Audiences, highlighting the platform's lean toward adult-oriented entertainment.

In conclusion, Netflix's library not only showcases global storytelling but also reflects evolving viewer preferences, with a clear dominance of movies and a growing demand for mature and diverse content.
"""
