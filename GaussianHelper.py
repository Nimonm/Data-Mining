from sklearn.mixture import GaussianMixture


class GaussianMixture2(GaussianMixture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = None

    def fit(self, X, y=None):
        self.X = X
        super().fit(X)

    @property
    def labels_(self):
        return self.predict(self.X)
