from load_data import load_data
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# load data
X, Y = load_data()

model = LogisticRegression()
rfe = RFE(model, 3, verbose=True)
fit = rfe.fit(X, Y)

# Number of selected features
print(fit.n_features_)

# Selected features are marked True
print(fit.support_)

# Selected features are marked 1
print(fit.ranking_)
