# importing package
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bar_plot_for_accuracy_comparison():
    positive_values = [.906, .91, .89, .899]
    negative_values = [.904, .95, .86, .91]
    index = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    df = pd.DataFrame({'LLM': positive_values,
                        'SVM': negative_values}, index=index)
    ax = df.plot.bar(rot=0, color={"LLM": "green", "SVM": "red"})
    ax.legend(["SVM", "LLM"], loc="upper right")
    plt.yticks([i/10 for i in range(0,12)])
    plt.title("Comparison of SVM and LLM")
    plt.show()

def line_chart_for_time_comparison():
    x = [x for x in range(0, 100000, 1000)]
    time_taken_for_svm = [0.0008*x for x in range(0, 100000, 1000)]
    time_taken_for_llm = [0.1*x for x in range(0, 100000, 1000)]
    plt.plot(x, time_taken_for_svm, label="SVM")
    plt.plot(x, time_taken_for_llm, label="LLM")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Time Taken (seconds)")
    plt.title("Speed Comparison of LLM and SVM")
    plt.legend()
    plt.show()

def dataset_star_split():
    df = pd.read_csv('../more_datasets/topics.csv', header=0,
                     names=['na1', 'na2', 'na3', 'na4', 'na5', 'na6', 'score', 'na7', 'summary', 'text'])
    overallCounts = {}
    # Calculate how many 5,4,3,2,1 star reviews we have
    for i in range(1, 6):
        size = len(df.query('score == ' + str(i)).index)
        overallCounts[str(i) + "*"] = size;
    fig, ax = plt.subplots()
    plt.title("Proportion of Reviews by Overall Rating")
    ax.pie(overallCounts.values(), labels=overallCounts.keys(), autopct='%1.1f%%')
    plt.show()
bar_plot_for_accuracy_comparison()