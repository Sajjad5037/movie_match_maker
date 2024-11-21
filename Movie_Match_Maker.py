import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Read user data from a CSV file
# The CSV should have columns: User, Movie, Rating
df = pd.read_csv('user_movie_ratings.csv')

# Transform the data into a dictionary for easier processing
movie_user_preferences = defaultdict(dict)
for _, row in df.iterrows():
    movie_user_preferences[row['User']][row['Movie']] = row['Rating']

# Function to calculate Euclidean distance-based similarity score
def sim_distance(prefs, person1, person2):
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
    if len(si) == 0: 
        return 0
    sum_of_squares = sum(pow(prefs[person1][item] - prefs[person2][item], 2) for item in si)
    return 1 / (1 + sum_of_squares)

# Generate recommendations for a user
def get_recommendations(prefs, target_user, similarity_func):
    totals = defaultdict(float)
    sim_sums = defaultdict(float)

    for other_user in prefs:
        if other_user == target_user:
            continue
        similarity = similarity_func(prefs, target_user, other_user)
        if similarity <= 0: 
            continue

        for item in prefs[other_user]:
            if item not in prefs[target_user] or prefs[target_user][item] == 0:
                totals[item] += prefs[other_user][item] * similarity
                sim_sums[item] += similarity

    # Create a sorted list of recommendations
    rankings = [(total / sim_sums[item], item) for item, total in totals.items() if sim_sums[item] != 0]
    rankings.sort(reverse=True)
    return rankings

# Generate similarity scores for all users
similarity_scores = {}
users = list(movie_user_preferences.keys())
for i in range(len(users)):
    for j in range(i + 1, len(users)):
        user1 = users[i]
        user2 = users[j]
        score = sim_distance(movie_user_preferences, user1, user2)
        similarity_scores[(user1, user2)] = score

# Save similarity scores to a CSV
similarity_df = pd.DataFrame(
    [(user1, user2, score) for (user1, user2), score in similarity_scores.items()],
    columns=['User1', 'User2', 'Similarity']
)
similarity_df.to_csv('similarity_scores.csv', index=False)

# Generate recommendations for each user
recommendations = {}
for user in movie_user_preferences.keys():
    recommendations[user] = get_recommendations(movie_user_preferences, user, sim_distance)

# Save recommendations to a CSV
recommendation_list = []
for user, recs in recommendations.items():
    for score, movie in recs:
        recommendation_list.append((user, movie, score))
recommendation_df = pd.DataFrame(recommendation_list, columns=['User', 'Movie', 'Score'])
recommendation_df.to_csv('recommendations.csv', index=False)

# Visualization: Plot ratings for two popular movies
data = []
popular_movies = ['Django Unchained', 'Avenger: Age of Ultron']
for user in movie_user_preferences.keys():
    try:
        data.append((
            user,
            movie_user_preferences[user][popular_movies[0]],
            movie_user_preferences[user][popular_movies[1]]
        ))
    except KeyError:
        pass  # Ignore users without ratings for the required movies

df_vis = pd.DataFrame(data, columns=['user', 'django', 'avenger'])

# Plot the ratings
plt.scatter(df_vis.django, df_vis.avenger)
plt.xlabel(f'{popular_movies[0]} Ratings')
plt.ylabel(f'{popular_movies[1]} Ratings')
for i, txt in enumerate(df_vis.user):
    plt.annotate(txt, (df_vis.django[i], df_vis.avenger[i]))
plt.title(f'User Ratings: {popular_movies[0]} vs {popular_movies[1]}')
plt.show()

# Example usage
example_recommendations = recommendations['Sam']
print(f"\nRecommendations for Sam: {example_recommendations}")
