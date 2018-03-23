from sklearn.ensemble import ExtraTreesClassifier
from load_data import load_data

X, Y = load_data()

model = ExtraTreesClassifier()
fit = model.fit(X, Y)

# The larger a score is, the more important the corresponding feature is
print(fit.feature_importances_)
