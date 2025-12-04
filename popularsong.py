
import pandas as pd
import numpy as np


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
                            BaggingClassifier, BaggingRegressor, \
                            GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, pair_confusion_matrix, r2_score, confusion_matrix, \
                            classification_report, precision_score, \
                            accuracy_score, roc_curve, roc_auc_score

import scikitplot as skplt


from dmba import regressionSummary, exhaustive_search
from dmba import adjusted_r2_score, AIC_score, BIC_score
from dmba import classificationSummary, gainsChart, liftChart, plotDecisionTree

# Visualization and aesthetics
import matplotlib.pylab as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2,'font.family': [u'times']})
plt.style.use('seaborn-v0_8-whitegrid')
plt.rc('text', usetex = False)
plt.rc('font', family = 'serif')
plt.rc('xtick', labelsize = 10)
plt.rc('ytick', labelsize = 10)
plt.rc('font', size = 12)
plt.rc('figure', figsize = (6, 5))

#for plotting decision trees
import pydotplus as pplus
import graphviz
from IPython.display import Image
from six import StringIO



from sklearn.tree import export_graphviz



dfSpotify  = pd.read_csv("spotify_2024.csv", encoding="latin1")
dfSpotify.head()
# Using encoding to make sure the file is read properly

dfSpotify.columns

def print_tree(estimator, features, class_names=None, filled=True):
    tree = estimator
    names = features
    color = filled
    classn = class_names

    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, feature_names=features,\
                    class_names=classn, filled=filled)
    graph = pplus.graph_from_dot_data(dot_data.getvalue())
    return(graph)

"""Cleaning Data"""

missing_values = dfSpotify.isnull().sum()

import numpy as np

def clean_numeric_column(df, column_name):
    if column_name in df.columns:

        df[column_name] = pd.to_numeric(df[column_name].astype(str).str.replace(',', ''), errors='coerce')


        if df[column_name].isnull().any():
            median_val = df[column_name].median()
            df[column_name] = df[column_name].fillna(median_val)


        if not df[column_name].isnull().any() and not np.isinf(df[column_name]).any():

             if df[column_name].astype(float).apply(lambda x: x.is_integer()).all():
                df[column_name] = df[column_name].astype(int)
    return df

columns_to_clean = ['All Time Rank','Spotify Streams','Spotify Playlist Count','Spotify Playlist Reach','YouTube Views','YouTube Likes','TikTok Posts','TikTok Likes','TikTok Views','YouTube Playlist Reach',
    'AirPlay Spins','SiriusXM Spins','Deezer Playlist Count','Deezer Playlist Reach','Amazon Playlist Count','Pandora Streams','Pandora Track Stations','Soundcloud Streams','Shazam Counts','TIDAL Popularity'
]
# Cleaning columns to ensure models present accurately

for col in columns_to_clean:
    dfSpotify = clean_numeric_column(dfSpotify, col)



dfSpotify['Explicit Track'] = dfSpotify['Explicit Track'].astype(int)
#Turning explicit track into a binary

dfSpotify.head()

dfSpotify.info()

"""Training

Top Hit variable is going to be if its in the first 100
"""

dfSpotify['Is Top Hit'] = (dfSpotify['All Time Rank'] <= 100).astype(int)

columns_to_exclude = [
    'Track',
    'Album Name',
    'ISRC',
    'Artist',
    'Release Date',
    'Track Score',
    'All Time Rank',
    'Is Top Hit'
]
# Dropped track, album, and IRSC becuase they had over 4000 unique values each.

Xvar = dfSpotify.drop(columns = columns_to_exclude)
yvar = dfSpotify['Is Top Hit']

print(Xvar.columns)

X2_train, X2_test, Y2_train, Y2_test = train_test_split(Xvar, yvar, test_size=0.25, random_state=7)
print(X2_train.shape, Y2_train.shape)
print(X2_test.shape, Y2_test.shape)

"""Decision Tree


"""

features = Xvar.columns

class_labels = ['Not Top Hit (0)', 'Top Hit (1)']

DT_spotify = DecisionTreeClassifier(max_depth=4, max_leaf_nodes=None, \
                                      max_features=None, random_state=23)
# Changing max depth for optimization

DT_spotify.fit(X2_train, Y2_train)

"""Top Hit 1 = Was in the in the top 100 of songs for all time rank


Top Hit (0) = Wasn't in the top 100 of songs for all time rank
"""

plotDecisionTree(DT_spotify, feature_names=features, class_names=class_labels)

plt.savefig('decision_tree_plot.png')

DT_prediction_ctr=DT_spotify.predict(X2_train)

cm_DTC = confusion_matrix(Y2_train, DT_prediction_ctr)


skplt.metrics.plot_confusion_matrix(Y2_train, DT_prediction_ctr, figsize=(4,4), cmap="PuOr")

plt.savefig('confusion_matrix_plot.png')

"""True Negative: 3366 songs were predicted to not be Top Hits correctly

True Positives: 39 songs were predicted correcty by the model to be Top Hits

False Positives: 2 Songs were predicted to be false positives which was inaccurate

False Negatives: Model inccorectly predicted that 39 songs were top hits
"""
DT_prediction_cte = DT_spotify.predict(X2_test)

#Printing classification port to teminal
print(classification_report(Y2_test, DT_prediction_cte, target_names=class_labels))

#Calculating accuracy for predicitions
accuracy = accuracy_score(Y2_test, DT_prediction_cte)
print(f"Accuracy (Test Set): {accuracy:.4f}")

#AUC & ROC scores
DT_proba_cte = DT_spotify.predict_proba(X2_test)
roc_auc = roc_auc_score(Y2_test, DT_proba_cte[:, 1])
print(f"AUC-ROC Score (Test Set): {roc_auc:.4f}")