from sklearn.decomposition import PCA
from load_data import load_data

X, Y = load_data()

pca = PCA(n_components=3)
fit = pca.fit(X, Y)

# Dimensionality reduced data bears little resemblance to the actual data
print(fit.explained_variance_ratio_)
print(fit.components_)
