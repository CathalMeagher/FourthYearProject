from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

train_times = [39.69709610939026, 39.56657600402832, 39.92631697654724, 40.223220109939575, 41.205652952194214]
prediction_times = [5.644813060760498, 5.616560935974121, 5.59918212890625, 5.601227045059204, 5.602724075317383]
processing_times = [95.88]

# creating the dataset
data = {'Text Pre-Processing': processing_times[0], 'Training': mean(train_times), 'Prediction': mean(prediction_times)}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(courses, values,
        width=0.4)

plt.ylabel("Time Taken (s)")
plt.title("Time Taken for Model Training, Prediction and Text Pre-Processing ")
plt.show()