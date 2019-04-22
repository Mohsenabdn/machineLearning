from sklearn import linear_model
from sklearn.metrics import roc_auc_score


def auc(variables, target, df):
    # Evaluate area under the receiver operating characteristic curve (AUROC)
    # Input : variables and target are list type and df is the DataFrame\
    # including variables and target as its columns
    # Output : AUROC as a real number between 0 and 1 (1 is an ideal case)

    x = df[variables]
    y = df[target]

    log_reg = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
    log_reg.fit(x, y)

    predictions = log_reg.predict_proba(x)[:, 1]
    return roc_auc_score(y, predictions)


def best_next(current_vars, candidate_vars, target, df):
    # Find the best variable in combination of the previous variables for the model
    # Input : current_vars, condidate_vars, and target are list type and df is the\
    # DataFrame including them as its columns
    # Output : The best variable to add to the previous ones

    best_var = None
    best_auc = -1

    for v in candidate_vars:
        auc_v = auc(current_vars + [v], target, df)

        if auc_v > best_auc:
            best_auc = auc_v
            best_var = v

    return best_var
