import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import metrics
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib.pyplot import savefig
import shap

filename = 'C:/Users/XXX.csv'
data0 = read_csv(filename, header=0)
X = data0.iloc[:, :XX]
y = data0['XXX']
test_size = 0.10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=48)
print(X_train.shape, y_train.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

param_grid = {'n_estimators': [36, 56, 64, 82, 128, ],
              'learning_rate': [0.1, 0.2, ],
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': [2, 4, 6, 8, ],
              'subsample': [0.9, 0.8, 1], #
              'min_samples_split': [1, 2, 3, ],#
              'min_samples_leaf': [1, 2, 3, ]}
n_splits = 10
seed = 10
k_fold = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
gb_regressor = GradientBoostingRegressor()
model = GridSearchCV(estimator=gb_regressor, scoring='r2', param_grid=param_grid, cv=k_fold,
                     verbose=3, n_jobs=-1, return_train_score=True)
model.fit(X_train_std, y_train)
print(model.cv_results_)
df_result = pd.DataFrame(model.cv_results_)
df_result.to_csv('C:/Users/XXX.csv')
best_model = model.best_estimator_
joblib.dump(best_model, 'C:/Users/XXX.pkl')
joblib.dump(scaler, 'C:/Users/XXX.pkl')

y_train_hat = best_model.predict(X_train_std)
y_test_hat = best_model.predict(X_test_std)
X_train_original = pd.DataFrame(X_train.values, columns=X.columns)
feature_names = X.columns
X_train_std_df = pd.DataFrame(X_train_std, columns=feature_names)
explainer = shap.TreeExplainer(best_model)
shap_values_train = explainer.shap_values(X_train_std_df)
shap.summary_plot(shap_values_train, X_train_std_df, show=False)
fontsize = 12
plt.figure(figsize=(3.5, 3))
plt.style.use('default')
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.tick_params(direction='in')
plt.title(('Train RMSE: {:.2e}'.format(np.sqrt(metrics.mean_squared_error(y_train, y_train_hat))),
           'Test RMSE: {:.2e}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_test_hat)))), fontsize=fontsize)
plt.legend((a, b), ('Train', 'Test'), fontsize=fontsize, handletextpad=0.1, borderpad=0.1)
plt.tight_layout()
savefig("C:/Users/XXX.jpg", bbox_inches='tight')
np.savetxt('C:/Users/XXX.csv', y_test_hat, delimiter=',')
np.savetxt('C:/Users/XXX.csv', y_test, delimiter=',')
np.savetxt('C:/Users/XXX.csv', y_train_hat, delimiter=',')
np.savetxt('C:/Users/XXX.csv', y_train, delimiter=',')
plt.show()
