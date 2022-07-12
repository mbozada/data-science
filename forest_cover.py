import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import auc, confusion_matrix, ConfusionMatrixDisplay, classification_report, make_scorer, roc_auc_score, f1_score

# Models
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB, GaussianNB


COLUMNS = [
    # Quantitative Features
    "elevation", # Elevation in meters
    "azimuth", # Aspect in degrees azimuth
    "slope", # Slope in degrees
    "horz_water", # Horizontal distance to nearest surface water features in meters
    "vert_water", # Vertical distance to nearest surface water features in meters
    "horz_roads", # Horizontal distance to nearest roadway in meters
    "hillshade_9am", # Hillshade index at 9am on the Summer Solstice 0-255 index
    "hillshade_noon", # Hillshade index at noon on the Summer Solstice 0-255 index
    "hillshade_3pm", # Hillshade index at 3pm on the Summer Solstice 0-255 index
    "horz_fire", # Horizontal distance to nearest wildfire ignition points in meters

    # Qualitative Features
    # Wilderness area designation given as 4 binary columns
    "wa1","wa2","wa3","wa4",
    # Soil Type given as 40 binary columns
    'st1', 'st2', 'st3', 'st4', 'st5', 'st6', 'st7', 'st8', 'st9', 'st10',
    'st11', 'st12', 'st13', 'st14', 'st15','st16', 'st17', 'st18', 'st19', 'st20',
    'st21', 'st22', 'st23', 'st24', 'st25', 'st26', 'st27', 'st28', 'st29','st30',
    'st31', 'st32', 'st33', 'st34', 'st35', 'st36', 'st37', 'st38', 'st39', 'st40',

    # Classes
    "cover_type", # Integer 1-7 Cover Type Designation
]

TARGET_NAMES = {
    1:"Spruce/Fir",
    2:"Lodgepole Pine",
    3:"Ponderosa Pine",
    4:"Cottonwood/Willow",
    5:"Aspen",
    6:"Douglas-fir",
    7:"Krummholz"
}

if __name__ == "__main__":
    
    df = pd.read_csv("./forest_cover.csv", names=COLUMNS)
    df.head()


    feature_cols = COLUMNS[:-1]
    X = df.loc[:, feature_cols]
    y = df.cover_type

    lda_clf = Pipeline(steps=[
        ('preprocesser', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis()),
    ])

    gnb_clf = Pipeline(steps=[
        ('preprocesser', StandardScaler()),
        ('clf', GaussianNB()),
    ])

    scoring = [
        'roc_auc_ovr',
        'f1_weighted',
    ]

    lda_cross = cross_validate(lda_clf, X, y, cv=10, scoring=scoring)
    gnb_cross = cross_validate(gnb_clf, X, y, cv=10, scoring=scoring)

    print(lda_cross)

    print(gnb_cross)