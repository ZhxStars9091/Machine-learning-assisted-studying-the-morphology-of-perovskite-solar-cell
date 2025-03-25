import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import metrics
from scipy import stats
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
import shap
from matplotlib.pyplot import savefig

filename = 'C:/Users/XXX.csv'
data0 = read_csv(filename)
X = data0[['XXX', 'XXX', 'XXX',]]
y = data0[['XX', 'XX']]
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=64)
print(X_train.shape, y_train.shape)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
param_grid = {
    'estimator__n_estimators': [32, 50, 64, 100, 128, 150, 168],
    'estimator__learning_rate': [0.01, 0.1, 0.3, 0.5],
    'estimator__loss': ['linear', 'square', 'exponential']
}

n_splits = 10
seed = 16
k_fold = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
base_regressor = DecisionTreeRegressor(max_depth=6)
ada_boost_regressor = AdaBoostRegressor(base_regressor, random_state=seed)
multi_output_regressor = MultiOutputRegressor(ada_boost_regressor)
model = GridSearchCV(estimator=multi_output_regressor, scoring='r2', param_grid=param_grid, cv=k_fold,
                     verbose=3, n_jobs=-1, return_train_score=True)
model.fit(X_train_std, y_train)
print(model.cv_results_)
df_result = pd.DataFrame(model.cv_results_)
df_result.to_csv('C:/Users/XXX.csv', index=False)
best_model = model.best_estimator_
joblib.dump(best_model, 'C:/Users/XXX.pkl')
joblib.dump(scaler, 'C:/Users/XXX.pkl')
y_train_hat = model.predict(X_train_std)
y_test_hat = model.predict(X_test_std)
fontsize = 12
plt.figure(figsize=(6, 3))
plt.style.use('default')
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rcParams['font.family'] = "Arial"
plt.subplot(1, 2, 1)
a = plt.scatter(y_train['XX'], y_train_hat[:, 0],)
plt.plot([y_train['XX'].min(), y_train['XX'].max()], [y_train['XX'].min(), y_train['XX'].max()], )
plt.xlabel('Observation (XX)', fontsize=fontsize)
plt.ylabel('Prediction (XX)', fontsize=fontsize)
plt.tick_params(direction='in')
plt.title(('Train RMSE: {:.2e}'.format(np.sqrt(metrics.mean_squared_error(y_train['XX'], y_train_hat[:, 0]))),
           'Test RMSE: {:.2e}'.format(np.sqrt(metrics.mean_squared_error(y_test['XX'], y_test_hat[:, 0])))),
          fontsize=fontsize)
b = plt.scatter(y_test['XX'], y_test_hat[:, 0],)
plt.legend((a, b), ('Train', 'Test'), fontsize=fontsize, handletextpad=0.1, borderpad=0.1)
plt.subplot(1, 2, 2)
a = plt.scatter(y_train['XX'], y_train_hat[:, 1],)
plt.plot([y_train['XX'].min(), y_train['XX'].max()], [y_train['XX'].min(), y_train['XX'].max()],)
plt.xlabel('Observation (XX)', fontsize=fontsize)
plt.ylabel('Prediction (XX)', fontsize=fontsize)
plt.tick_params(direction='in')
plt.title(('Train RMSE: {:.2e}'.format(np.sqrt(metrics.mean_squared_error(y_train['XX'], y_train_hat[:, 1]))),
           'Test RMSE: {:.2e}'.format(np.sqrt(metrics.mean_squared_error(y_test['XX'], y_test_hat[:, 1])))),
          fontsize=fontsize)
b = plt.scatter(y_test['XX'], y_test_hat[:, 1],)
plt.legend((a, b), ('Train', 'Test'), fontsize=fontsize, handletextpad=0.1, borderpad=0.1)
plt.tight_layout()
savefig("C:/Users/XX.jpg", bbox_inches='tight')
plt.show()
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
best_multi_output_model = model.best_estimator_
for i, target_name in enumerate(['FF', 'PCE']):
    print(f"解释目标变量 {target_name} 的模型")
    best_single_model = best_multi_output_model.estimators_[i]
    explainer = shap.KernelExplainer(best_single_model.predict, X_test_std)
    shap_values = explainer.shap_values(X_test_std)
    shap.summary_plot(shap_values, X_test_std, feature_names=X.columns, plot_type='bar')
    shap.summary_plot(shap_values, X_test_std, feature_names=X.columns)
best_multi_output_model = model.best_estimator_
sample_idx = 0
sample = X_test_std[sample_idx:sample_idx + 1]
figure_path = "C:/Users/XX/"
for i, target_name in enumerate(['XX', 'XX']):
    print(f"解释目标变量 {target_name} 的模型")
    best_single_model = best_multi_output_model.estimators_[i]
    explainer = shap.KernelExplainer(best_single_model.predict, X_test_std)
    shap_values = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=X.columns, plot_type='bar', show=False)
    savefig(f"{figure_path}AdaBoost_{target_name}_global_shap.jpg", bbox_inches='tight',dpi=600)
    plt.close()
    plt.figure()
    shap.summary_plot(shap_values, sample, feature_names=X.columns, show=False)
    savefig(f"{figure_path}AdaBoost_{target_name}_summary_shap.jpg", bbox_inches='tight',dpi=600)
    plt.close()
    plt.figure()
    shap.force_plot(explainer.expected_value,
                    shap_values,
                    sample,
                    feature_names=X.columns,
                    matplotlib=True,
                    show=False,
                    text_rotation=15)
    plt.title(f"Sample {sample_idx} - {target_name} SHAP Values", fontsize=fontsize)
    savefig(f"{figure_path}AdaBoost_{target_name}_sample_{sample_idx}_force_plot.jpg", bbox_inches='tight',dpi=600)
    plt.close()
    plt.figure()
    shap.decision_plot(explainer.expected_value,
                       shap_values,
                       sample,
                       feature_names=X.columns.tolist(),
                       show=False)
    savefig(f"{figure_path}AdaBoost_{target_name}_sample_{sample_idx}_decision_plot.jpg", bbox_inches='tight',dpi=600)
    sample_df = pd.DataFrame(sample, columns=X.columns)
    sample_df.to_csv(f"{figure_path}AdaBoost_{target_name}_sample_{sample_idx}_data.csv", index=False)

