import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score
import optuna



def pregame_preds():

    df = pd.read_csv('soccermatches.csv')

    df['AGPG_TY'] = df['AG_TY'] / df['AM_TY']
    df['HGPG_TY'] = df['HG_TY'] / df['HM_TY']
    df['AGPG_LY'] = df['AG_LY'] / df['AM_LY']
    df['HGPG_LY'] = df['HG_LY'] / df['HM_LY']

    df['AGD_TY_LY'] = df['AG_TY'] - df['AG_LY']
    df['HGD_TY_LY'] = df['HG_TY'] - df['HG_LY']

    features_to_interact = [
        "HG_LY","HGA_LY","HM_LY","AG_LY","AGA_LY","AM_LY",
        "HG_TY","HGA_TY","HM_TY","AG_TY","AGA_TY","AM_TY"
    ]

    for feature1, feature2 in combinations(features_to_interact, 2):
        new_col_name = f"{feature1}_x_{feature2}"
        df[new_col_name] = df[feature1] * df[feature2]

    params = {
        'objective' : 'multi:softprob',
        'num_class' : 3,
        'learning_rate' : 0.05,
        'max_depth' : 3,
        #'n_estimators' : 250,
        'subsample' : 0.8
    }

    X = df.drop(columns=["Hwins","Awins","IsDraw","goals2H","Hgoals","Agoals","Hgoals1H","Agoals1H"])

    y_extract = df[['Hwins', 'Awins', 'IsDraw']]
    label_map = {'Hwins': 0, 'Awins': 1, 'IsDraw': 2}

    y = y_extract.idxmax(axis=1).map(label_map)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    dtrain = xgb.DMatrix(X, label=y)

    bst = xgb.train(params, dtrain)

    df2 = pd.read_csv('soccermatches2.csv')

    X_test = df2.drop(columns=["Hwins","Awins","IsDraw","goals2H","Hgoals","Agoals","Hgoals1H","Agoals1H"])
    y_test_extract = df2[['Hwins', 'Awins', 'IsDraw']]
    y_test = y_test_extract.idxmax(axis=1).map(label_map)

    dtest = xgb.DMatrix(X_test)
    y_pred_probs = bst.predict(dtest)

    print("Predicted Probabilities (HWin, AWin, Draw) for first 5 games:")
    print(y_pred_probs[:5])

    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred_labels)
    print(f"\nAccuracy: {accuracy:.4f}")


    df2.insert(loc=0, column='hProb', value=[y_pred_probs[i][0] for i in range(len(y_pred_probs))])
    df2.insert(loc=1, column='dProb', value=[y_pred_probs[i][1] for i in range(len(y_pred_probs))])
    df2.insert(loc=2, column='aProb', value=[y_pred_probs[i][2] for i in range(len(y_pred_probs))])

    return df2

def halftime_preds(out_df):

    df = pd.read_csv('soccermatches.csv')

    params = {
        'objective' : 'multi:softprob',
        'num_class' : 3,
        'learning_rate' : 0.2, 
        'max_depth' : 3, 
        #'n_estimators' : 100, 
        'subsample' : 0.8
    }

    X = df.drop(columns=["Hwins","Awins","IsDraw","goals2H","Hgoals","Agoals"])

    y_extract = df[['Hwins', 'Awins', 'IsDraw']]
    label_map = {'Hwins': 0, 'Awins': 1, 'IsDraw': 2}

    y = y_extract.idxmax(axis=1).map(label_map)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    dtrain = xgb.DMatrix(X, label=y)

    bst = xgb.train(params, dtrain)

    df2 = pd.read_csv('soccermatches2.csv')

    X_test = df2.drop(columns=["Hwins","Awins","IsDraw","goals2H","Hgoals","Agoals"])
    y_test_extract = df2[['Hwins', 'Awins', 'IsDraw']]
    y_test = y_test_extract.idxmax(axis=1).map(label_map)

    dtest = xgb.DMatrix(X_test)
    y_pred_probs = bst.predict(dtest)

    print("Predicted Halftime Probabilities (HWin, AWin, Draw) for first 5 games:")
    print(y_pred_probs[:5])

    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred_labels)
    print(f"\nAccuracy: {accuracy:.4f}")


    out_df.insert(loc=0, column='hProbHT', value=[y_pred_probs[i][0] for i in range(len(y_pred_probs))])
    out_df.insert(loc=1, column='dProbHT', value=[y_pred_probs[i][1] for i in range(len(y_pred_probs))])
    out_df.insert(loc=2, column='aProbHT', value=[y_pred_probs[i][2] for i in range(len(y_pred_probs))])
    
    return out_df

def num_goals_preds_old(out_df):

    df = pd.read_csv('soccermatches.csv')

    params = {
        'objective' : 'multi:softprob',
        'num_class' : 15
    }

    X = df.drop(columns=["Hwins","Awins","IsDraw","goals2H","Hgoals","Agoals","Hgoals1H","Agoals1H"])

    y = df['Hgoals'] + df['Agoals']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    dtrain = xgb.DMatrix(X, label=y)

    bst = xgb.train(params, dtrain)

    df2 = pd.read_csv('soccermatches2.csv')

    X_test = df2.drop(columns=["Hwins","Awins","IsDraw","goals2H","Hgoals","Agoals","Hgoals1H","Agoals1H"])
    y_test = df2['Hgoals'] + df2['Agoals']

    dtest = xgb.DMatrix(X_test)
    y_pred_probs = bst.predict(dtest)


    vector = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    

    y_pred_evs = [np.dot(np.array(y_pred_probs[i]), vector) for i in range(len(y_pred_probs))]

    mae = mean_absolute_error(np.array(y_test), np.array(y_pred_evs))
    print(f"MAE: {mae:.4f}")


    out_df.insert(loc=0, column='predGoals2H', value=y_pred_evs)
    out_df.to_csv('pred.csv', index=False)


def num_goals_preds(out_df):
    df = pd.read_csv('soccermatches.csv')

    X_train = df.drop(columns=["Hwins","Awins","IsDraw","goals2H","Hgoals","Agoals","Hgoals1H","Agoals1H"])
    y_train = df['Agoals'] + df['Hgoals']

    imputer = SimpleImputer(strategy='mean')

    X_train_imputed = imputer.fit_transform(X_train)

    df2 = pd.read_csv('soccermatches2.csv')

    X_test = df2.drop(columns=["Hwins","Awins","IsDraw","goals2H","Hgoals","Agoals","Hgoals1H","Agoals1H"])
    y_test = df2['Agoals'] + df2['Hgoals']


    X_test_imputed = imputer.transform(X_test)

    model = PoissonRegressor(max_iter=10000)

    model.fit(X_train_imputed, y_train)

    y_pred = model.predict(X_test_imputed)

    mae = mean_absolute_error(np.array(y_test), np.array(y_pred))
    print(f"MAE: {mae:.4f}")

    out_df.insert(loc=0, column='predGoals2H', value=y_pred)
    out_df.to_csv('pred.csv', index=False)





def num_goals_preds_new():

    df = pd.read_csv('soccermatches.csv')
    
    drop_cols = ["Hwins","Awins","IsDraw","goals2H","Hgoals","Agoals","Hgoals1H","Agoals1H"]
    X = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=np.number)
    y = df['Hgoals'] + df['Agoals']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        learning_rate=0.05,
        random_state=42,
        n_estimators=100,
        subsample=0.8
    )
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae:.4f}")

    r2 = r2_score(y_test, y_pred)
    print(f"R2: {r2}")


    #out_df.insert(loc=0, column='predGoals', value=y_pred)
    #out_df.to_csv('pred.csv', index=False)




def num_goals_preds_optuna():
    # 1. Load and prepare the full dataset
    df = pd.read_csv('soccermatches.csv')
    df['AGPG_TY'] = df['AG_TY'] / df['AM_TY']
    df['HGPG_TY'] = df['HG_TY'] / df['HM_TY']
    df['AGPG_LY'] = df['AG_LY'] / df['AM_LY']
    df['HGPG_LY'] = df['HG_LY'] / df['HM_LY']

    df['AGD_TY_LY'] = df['AG_TY'] - df['AG_LY']
    df['HGD_TY_LY'] = df['HG_TY'] - df['HG_LY']

    drop_cols = ["Hwins","Awins","IsDraw","goals2H","Hgoals","Agoals","Hgoals1H","Agoals1H"]
    X = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=np.number)
    y = df['Hgoals'] + df['Agoals']


    # Split data once into a training set for tuning and a final test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define the objective function for Optuna
    def objective(trial):
        """
        This function takes a trial, suggests hyperparameters,
        trains a model, and returns its cross-validated MAE.
        """
        # Define the search space for hyperparameters
        params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True) # L2 regularization
        }

        model = xgb.XGBRegressor(**params)

        # Evaluate the model using cross-validation
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        print(model.feature_importances_)
        # Optuna minimizes, so we return the positive MAE
        return -score.mean()

    # 3. Create and run the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50) # Run 50 trials to find the best params

    print("Study finished!")
    print(f"Best trial's MAE: {study.best_value:.4f}")
    print("Best hyperparameters found:")
    print(study.best_params)
    
    # 4. Train the final model with the best hyperparameters
    best_params = study.best_params
    final_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    
    final_model.fit(X_train, y_train)

    # 5. Evaluate the final model on the unseen test set
    y_pred = final_model.predict(X_test)
    final_mae = mean_absolute_error(y_test, y_pred)
    final_r2 = r2_score(y_test, y_pred)

    print("\n--- Final Model Performance on Test Set ---")
    print(f"MAE: {final_mae:.4f}")
    print(f"R2 Score: {final_r2:.4f}")


# Run the tuning process


if __name__ == "__main__":
    #out_df = pregame_preds()
    #out_df = halftime_preds(out_df)
    num_goals_preds_optuna()

    # num_goals_preds_new()