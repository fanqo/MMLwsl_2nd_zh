import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# 可从https://archive.ics.uci.edu/ml/datasets/Internet+Advertisements下载
df = pd.read_csv('ad.data', header = None)

explanatory_variable_columns = set(df.columns.values)
explanatory_variable_columns.remove(len(df.columns.values) - 1)
response_variable_column = df[len(df.columns.values) - 1]
# 最后一列包含响应变量

y = [1 if e == 'ad.' else 0 for e in response_variable_column]
X = df[list(explanatory_variable_columns)].copy()
X.replace(to_replace = ' *\?', value = -1, regex = True, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

pipeline = Pipeline([
    ('clf', DecisionTreeClassifier(criterion = 'entropy'))
])
parameters = {
    'clf__max_depth': (150, 155, 160),
    'clf__min_samples_split': (2, 3),
    'clf__min_samples_leaf': (1, 2, 3)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1,
                           scoring='f1')
grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_estimator_.get_params()
print('最好的打分： {:0.3f}'.format(grid_search.best_score_))
print('最好的参数集：')
for para_name in sorted(parameters.keys()):
    print('\t{}: {}'.format(para_name, best_parameters[para_name]))

predictions = grid_search.predict(X_test)
print(classification_report(y_test, predictions))
