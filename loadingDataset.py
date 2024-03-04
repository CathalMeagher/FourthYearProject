# Datasets
import gzip
import pandas as pd
from matplotlib import pyplot as plt

datasets = ["dataset/food.json.gz", "dataset/industrialAndScientific.json.gz", "dataset/luxuryAndBeauty.json.gz"]

import json

productReviews = []

for file in datasets:
    g = gzip.open(file, 'r')
    for line in g:
        review = json.loads(line)
        productReviews.append(review)
df = pd.DataFrame(productReviews)

# We only care about reviewText, overall, summary
df = df[['reviewText', 'overall']]

# Drop any reviews with null values
df = df.dropna()
overallCounts = {}
# Calculate how many 5,4,3,2,1 star reviews we have
for i in range(1, 6):
    size = len(df.query('overall == ' + str(i)).index)
    overallCounts[str(i) + "*"] = size;

# Plot number of reviews from each category
fig, ax = plt.subplots()
plt.title("Proportion of Reviews by Overall Rating Before Selection")
ax.pie(overallCounts.values(), labels=overallCounts.keys(), autopct='%1.1f%%')
plt.show()

# Take the first 5000 of each star rating to give a uniform distribution of items
sortedProductReviews = []
for i in range(1, 6):
    firstThousands = df.query('overall == ' + str(i)).iloc[:5000]
    if i <= 2:
        firstThousands['sentiment'] = 'negative'
    elif i == 3:
        continue;
    else:
        firstThousands['sentiment'] = 'positive'

    sortedProductReviews.append(firstThousands)
sortedProductReviews = pd.concat(sortedProductReviews)

overallCounts = {}
# Calculate how many 5,4,3,2,1 star reviews we have
for i in range(1, 6):
    if i == 3: continue
    size = len(sortedProductReviews.query('overall == ' + str(i)).index)
    overallCounts[str(i) + "*"] = size;

# Plot number of reviews from each category
fig, ax = plt.subplots()
plt.title("Proportion of Reviews by Overall Rating After Selection")
ax.pie(overallCounts.values(), labels=overallCounts.keys(), autopct='%1.1f%%')
plt.show()

# Shuffle the dataframe so that reviews of same score are not beside each other
sortedProductReviews = sortedProductReviews.sample(frac=1).reset_index(drop=True)

sortedProductReviews.to_csv('base_data.csv', header=True, sep=',', index=False)
