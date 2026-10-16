"""Fixed paired monthly Ridge forecasts through the Qlib Model interface."""
import numpy as np
import pandas as pd
from qlib.model.base import Model
from sklearn.linear_model import Ridge


class MonthlyValuationRidge(Model):
    def __init__(self, features, targets, *, alpha=10., minimum_label_std=1e-8):
        self.features = list(features)
        self.targets = list(targets)
        self.alpha = alpha
        self.minimum_label_std = minimum_label_std

    def transform(self, frame):
        values = frame[self.features].replace([np.inf, -np.inf], np.nan)
        values = values.fillna(self.medians)
        values = (values-self.means)/self.scales
        # An entirely absent training coordinate cannot acquire a meaning from
        # the test set. Its explicit missingness flag is a different coordinate.
        values.loc[:, ~self.active] = 0.
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError('nonfinite transformed monthly feature')
        return values.to_numpy(float)

    def fit(self, dataset, **kwargs):
        data = dataset.prepare('train', col_set=['feature', 'label'])
        x = data['feature'][self.features].replace([np.inf, -np.inf], np.nan)
        y = data['label'][self.targets]
        if not np.isfinite(y.to_numpy()).all():
            raise ValueError('unknown training label')
        self.active = x.notna().any()
        self.medians = x.median().fillna(0.)
        filled = x.fillna(self.medians)
        self.means = filled.mean()
        self.scales = filled.std(ddof=0).replace(0., 1.)
        self.label_means = y.mean()
        self.label_scales = y.std(ddof=0)
        if (self.label_scales <= self.minimum_label_std).any():
            raise ValueError('degenerate training label variance')
        self.regressor = Ridge(alpha=self.alpha, fit_intercept=True)
        self.regressor.fit(self.transform(x), ((y-self.label_means)/self.label_scales).to_numpy(float))
        self.audit = {'training_rows': len(x), 'inactive_features': list(self.active.index[~self.active]),
                      'alpha': self.alpha, 'target_weights': [.5, .5]}

    def predict(self, dataset, segment='test'):
        x = dataset.prepare(segment, col_set='feature')
        y = self.regressor.predict(self.transform(x))
        y = y*self.label_scales.to_numpy()+self.label_means.to_numpy()
        return pd.DataFrame(y, index=x.index, columns=self.targets)
