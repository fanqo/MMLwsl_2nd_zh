import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, \
    confusion_matrix
from sklearn.pipeline import Pipeline

df = pd.read_csv('train.tsv.zip', compression='zip', delimiter='\t', header=0)
X, y = df['Phrase'], df['Sentiment'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])

parameters = {
    'vect__max_df': (0.25, 0.5),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'clf__C': (0.1, 1, 10)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs = 2,
                            verbose = 1, scoring = 'accuracy')
grid_search.fit(X_train, y_train)
print('最好的打分： {}'.format(grid_search.best_score_))
print('最好参数集：')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t{}: {}'.format(param_name, best_parameters[param_name]))




