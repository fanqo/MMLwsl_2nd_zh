import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])

parameters = {
    'vect__max_df': (0.25, 0.5, 0.75),
    'vect__stop_words': ('english', None),
    'vect__max_features': (2500, 5000, 10000, None),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'vect__norm': ('l1', 'l2'),
    'clf__penalty': ('l1', 'l2'),
    'clf__C': (0.01, 0.1, 1, 10)
}

df = pd.read_csv('./SMSSpamCollection', delimiter='\t',
                 names=['label', 'message'])

X = df['message'].values
y = df['label'].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

grid_search = GridSearchCV(pipeline, parameters, n_jobs=2,
                           verbose=1, scoring='accuracy', cv=3)
grid_search.fit(X_train, y_train)

print('最好的评分为： {:0.3f}'.format(grid_search.best_score_))
print('最好的参数集为：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t{}: {}'.format(param_name, best_parameters[param_name]))

predictions = grid_search.predict(X_test)
print('准确率： {}'.format(accuracy_score(y_test, predictions)))
print('精确率： {}'.format(precision_score(y_test, predictions)))
print('召回率： {}'.format(recall_score(y_test, predictions)))

#     
