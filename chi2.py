import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from load_data import load_data

# load data
X, Y = load_data()

# Chi-square test: It tests the dependency of two categorical variables
# => Check every pair of target and predictor variables and eliminate the irrelevant features
chi2test = SelectKBest(score_func=chi2, k=4)
fit = chi2test.fit(X, Y)

numpy.set_printoptions(precision=3)

# 4 important features are: test, age, plas, mass
print(fit.scores_)
features = fit.transform(X)

# View the first five rows of 4 important features
print(features[0:5, 0:4])
