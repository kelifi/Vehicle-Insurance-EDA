import lightgbm as lgb
import numpy as np
# import packages for hyperparameters tuning
from hyperopt import Trials, fmin, hp, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from preparation_data import x_train, y_train, x_test, y_test, cat_feat

random_state = 42
n_iter = 2
num_folds = 2
kf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)


def gb_mse_cv(params, random_state=random_state, cv=kf, X=x_train, y=y_train):
    # the function gets a set of variable parameters in "param"
    params = {'n_estimators': int(params['n_estimators']),
              'max_depth': int(params['max_depth']),
              'learning_rate': params['learning_rate'],
              'gamma': params['gamma'],
              'reg_alpha': params['reg_alpha'],
              'reg_lambda': params['reg_lambda'],
              'colsample_bytree': params['colsample_bytree'],
              'min_child_weight': params['min_child_weight']
              }

    # we use this params to create a new LGBM Regressor
    model = lgb.LGBMClassifier(random_state=42, **params)

    # and then conduct the cross validation with the same folds as before
    score = -cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    return score


# possible values of parameters
space = {'n_estimators': hp.quniform('n_estimators', 100, 200, 1),
         'max_depth': hp.quniform('max_depth', 2, 8, 1),
         'learning_rate': hp.loguniform("learning_rate", -4, -1),
         'gamma': hp.quniform('gamma', 0.1, 0.5, 0.1),
         'reg_alpha': hp.quniform('reg_alpha', 1.1, 1.5, 0.1),
         'reg_lambda': hp.uniform('reg_lambda', 1.1, 1.5),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.5),
         'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
         }

# trials will contain logging information
for column in cat_feat:
    x_train[column] = x_train[column].astype('int')
    x_test[column] = x_test[column].astype('int')

trials = Trials()

best = fmin(fn=gb_mse_cv,
            space=space,
            algo=tpe.suggest,  # optimization algorithm, hyperotp will select its parameters automatically
            max_evals=n_iter,  # maximum number of iterations
            trials=trials,  # logging
            )
# computing the score on the test set
model = lgb.LGBMClassifier(random_state=random_state, n_estimators=int(best['n_estimators']),
                           max_depth=int(best['max_depth']), learning_rate=best['learning_rate'], gamma=best['gamma'],
                           reg_alpha=best['reg_alpha'], reg_lambda=best['reg_lambda'],
                           colsample_bytree=best['colsample_bytree'],
                           min_child_weight=best['min_child_weight'])
model.fit(x_train, y_train)

preds = [pred[1] for pred in model.predict_proba(x_test)]
score = roc_auc_score(y_test, preds, average='weighted')
print(score)