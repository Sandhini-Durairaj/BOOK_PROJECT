import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from datetime import datetime

# --------------------------
# Dummy dataset
# --------------------------
data = {
    "num_pages": [100, 200, 300, 400, 150],
    "ratings_count": [50, 150, 200, 250, 100],
    "text_reviews_count": [10, 20, 30, 40, 15],
    "publication_date": ["2000-01-01", "2010-05-15", "2015-07-20", "2020-10-10", "2005-03-03"],
    "average_rating": [3.5, 4.0, 4.5, 5.0, 3.8]  # target
}

df = pd.DataFrame(data)

# Preprocess publication_date
df["publication_year"] = df["publication_date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").year)

# Features and target
X = df[["num_pages", "ratings_count", "text_reviews_count", "publication_year"]]
y = df["average_rating"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("ml_projects.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as ml_project.pkl")
