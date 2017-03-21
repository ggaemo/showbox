from datetime import datetime
import os
import pandas as pd
import re
import itertools
import collections
from sklearn import metrics
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import export_graphviz
from sklearn.base import clone

class Classification():

    def __init__(self, additional= None):
        print('running classification')
        self.data = None
        self.result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result')
        if additional:
            self.result_dir = os.path.join(self.result_dir, additional)

    def read_data(self, data, start_year):
        training_data = data
        def missing_value_imputation(data):
            print('missing value imputation')
            input_cols = [re.search("(.*)_-(\d)", col).group(1) for col in
                          data.columns if re.search("(.*)_-(""\d)",col)]
            unique_cols = np.unique(input_cols)
            for idx in data.index:
                for col in unique_cols:
                    to_align = [col + '_-{}'.format(i) for i in [1, 2, 3, 4, 5, 6]]
                    tmp = data.loc[idx, to_align]
                    null_idx = tmp.isnull()
                    if sum(null_idx) > 0 and sum(null_idx) <= 3:
                        X = np.where(~null_idx)[0]
                        X = X.reshape(len(X), 1)
                        y = tmp.iloc[np.where(~null_idx)[0]]
                        not_known = np.where(null_idx)[0]
                        not_known = not_known.reshape(len(not_known), 1)
                        clf = LinearRegression()
                        clf.fit(X, y)
                        impute = clf.predict(not_known)
                        impute = np.asarray([x if x > 0 else 0 for x in impute])

                        if len(impute) == 1:
                            impute = impute[0]
                        data.loc[idx, tmp.loc[null_idx].index] = impute
            return data

        if os.path.exists(os.path.join(os.path.dirname(os.getcwd()), 'result', str(start_year),
                                                    'training_data_관객수_imputation_{}.csv'.format(start_year))):
            training_data = pd.read_csv(
                os.path.join(os.path.dirname(os.getcwd()), 'result', str(start_year),
                             'training_data_관객수_imputation_{}.csv'.format(start_year)))
        else:
            training_data = missing_value_imputation(training_data)
            training_data.to_csv(
                os.path.join(os.path.dirname(os.getcwd()), 'result', str(start_year),
                             'training_data_관객수_imputation_{}.csv'.format(start_year)), index=False)

        opening_date = [datetime.strptime(re.search("\d{4}-\d{2}-\d{2}", x).group(), "%Y-%m-%d")
                                  for x in training_data.loc[:, "Identifier"]]
        opening_date = pd.Series(opening_date)

        validation_date = datetime(2015, 1, 1)
        test_date = datetime(2016, 1, 1)

        trn_idx = opening_date < validation_date
        val_idx = (opening_date >= validation_date) & (opening_date < test_date)
        test_idx = opening_date >= test_date

        trn_data = training_data.loc[trn_idx]
        val_data = training_data.loc[val_idx]
        test_data = training_data.loc[test_idx]

        def get_complete_data(data, week):
            columns = list(data.columns)
            for col in data.columns:
                match = re.search("_-(\d)", col)
                if match:
                    if int(match.group(1)) < week:
                        columns.remove(col)
            full_idx = data.loc[:,columns].notnull().all(axis=1)
            return data.loc[full_idx, columns]

        def make_training_data(data, week_range = (1,2,3,4,5,6)):
            data_by_week = dict()
            for week in week_range:
                data_by_week[week] = get_complete_data(data, week)
            return data_by_week

        trn_data = make_training_data(trn_data)
        val_data =  make_training_data(val_data)
        test_data =  make_training_data(test_data)
        print("read and split data")
        self.data = {"train" : trn_data, "validation" : val_data, "test" : test_data}

    def train_classifier(self, clf, param_grid, eval_metric, nationality, cond):

        def evaluate(clf, data, fit, nationality, cond=cond):
            data = data.loc[data["target"] > cond]

            if nationality in ["한국", "미국"]:
                data = data.loc[data["국적"] == nationality]

            X = data.drop(["target", "Identifier", "국적"], axis=1)
            y = data["target"]

            if fit:
                clf.fit(X, y)
            pred = clf.predict(X)
            score = collections.OrderedDict()
            mse = np.sqrt(metrics.mean_squared_error(y, pred))
            score["mse"] = mse
            mae = metrics.mean_absolute_error(y, pred)
            score["mae"] = mae

            def mape(y, mae):
                return np.mean([abs(a - b) / a * 100 for a, b in zip(y, pred)])

            score["mape"] = mape(y, mae)

            return [clf, score]

        def predict(clf, data):
            X = data.drop(["target", "Identifier", "국적"], axis=1)
            pred = [int(x) for x in clf.predict(X)]
            output = data.loc[:, ["Identifier", "target", "국적"]]
            output.loc[:, "prediction"] = pred
            return output

        def train(clf, week_range = (1,2,3,4,5,6), self=self, nationality = nationality):
            eval_columns = ["mse", "mae", "mape"]
            model_dict = dict()
            model_score = pd.DataFrame(columns=["val_mse", "test_mse", "val_mape", "test_mape"], index=week_range)
            clf_name = repr(clf).split("(")[0] ## print 되는걸 확인

            for week in week_range:
                print(clf_name, "week :" ,week)
                save_dir = os.path.join(self.result_dir, clf_name, str(week))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                trn_score = pd.DataFrame(columns=eval_columns)
                val_score = pd.DataFrame(columns=eval_columns)
                test_score = pd.DataFrame(columns=eval_columns)
                for params in itertools.product(*param_grid.values()):
                    tmp_param = {key: val for key, val in zip(param_grid.keys(), params)}
                    clf = clone(clf)
                    clf = clf.set_params(**tmp_param)
                    clf, trn_score.loc[repr(tmp_param)] = evaluate(clf, trn_data[week], True, nationality)

                    # clf, trn_score.loc[repr(tmp_param)] = evaluate(clf, pd.concat([trn_data[week], val_data[week],
                    #                                                               test_data[week]])
                    #                                                                   , True, nationality)

                    model_dict[repr(tmp_param)] = clf
                    _, val_score.loc[repr(tmp_param)] = evaluate(clf, val_data[week],
                                                                                 False,
                                                                                 nationality)
                    _, test_score.loc[repr(tmp_param)] = evaluate(clf, test_data[week], False, nationality)


                trn_score.sort_values(eval_metric, ascending=True, inplace=True)
                val_score.sort_values(eval_metric, ascending=True, inplace=True)
                test_score = test_score.loc[val_score.index]

                trn_score.to_csv(os.path.join(save_dir, "train_score.csv"))
                val_score.to_csv(os.path.join(save_dir, "validation_score.csv"))
                test_score.to_csv(os.path.join(save_dir, "test_score.csv"))

                model_score.loc[week, ["val_mse", "val_mape"]] =val_score.ix[0, ["mse", "mape"]].values

                model_score.loc[week, ["test_mse", "test_mape"]]  = test_score.loc[val_score.index[0], ["mse",
                                                                                                        "mape"]].values
                best_model = model_dict[val_score.index[0]]

                trn_prediction = predict(best_model, trn_data[week])
                val_prediction = predict(best_model, val_data[week])
                test_prediction = predict(best_model, test_data[week])
                agg_prediction = pd.concat([trn_prediction, val_prediction, test_prediction])


                if clf_name == 'DecisionTreeRegressor':
                    unique_vals = np.unique(trn_prediction["prediction"])
                    unique_vals = sorted(unique_vals, reverse = True)
                    for group, val in enumerate(unique_vals, start=1):
                        trn_prediction.loc[trn_prediction["prediction"] == val, "group"] = group
                        val_prediction.loc[val_prediction["prediction"] == val, "group"] = group
                        test_prediction.loc[test_prediction["prediction"] == val, "group"] = group


                trn_prediction.sort_values("target", inplace=True, ascending=False)
                val_prediction.sort_values("target", inplace=True, ascending=False)
                test_prediction.sort_values("target", inplace=True, ascending=False)
                val_test_prediction = pd.concat([val_prediction, test_prediction])
                val_test_prediction.sort_values("target", inplace=True, ascending=False)


                trn_prediction.to_csv(os.path.join(save_dir, "train_pred.csv"), index=False)
                val_prediction.to_csv(os.path.join(save_dir, "validation_pred.csv"), index=False)
                test_prediction.to_csv(os.path.join(save_dir, "test_pred.csv"), index=False)
                val_test_prediction.to_csv(os.path.join(save_dir, "val_test_pred.csv"), index=False)
                agg_prediction.to_csv(os.path.join(save_dir, "agg_pred.csv"), index=False)

                # over_30_val = val_data[week]val_data["target"] > 300000]
                # over_30_test = test_data.loc[test_data["target"] > 300000]
                #
                # over_50_val = val_prediction.loc[val_prediction["target"] > 500000]
                # over_50_test = test_prediction.loc[test_prediction["target"] > 500000]

                # over_50_val_kor = val_prediction.loc[(val_prediction["target"] > 500000)
                #                                  & (val_prediction["대표국적"] == "한국")]
                # over_50_test_kor = test_prediction.loc[(test_prediction["target"] > 500000)
                #                                    & (val_prediction["대표국적"] == "한국")]
                #
                # over_50_val_us = val_prediction.loc[(val_prediction["target"] > 500000)
                #                                  & (val_prediction["대표국적"] == "미국")]
                # over_50_test_us = test_prediction.loc[(test_prediction["target"] > 500000)
                #                                    & (val_prediction["대표국적"] == "미국")]

                ## export decision tree
                if clf_name == 'DecisionTreeRegressor':
                    print("make decision tree plot")
                    try:
                        export_graphviz(best_model, out_file="{}/best_classifier_{}.dot".format(save_dir, week),
                                        feature_names=trn_data[week].drop(["Identifier", "target", '국적'],
                                                                          axis=1).columns,
                                        filled=True, rounded=True, proportion=True
                                        )
                    except ValueError:
                        export_graphviz(best_model, out_file="{}/best_classifier_{}.dot".format(save_dir, week),
                                        feature_names=trn_data[week].drop(["Identifier", "target", '국적'],
                                                                          axis=1).columns,
                                        rounded=True, proportion=True
                                        )


                    os.system("dot -Tpng {0}/best_classifier_{1}.dot -o {0}/decision_tree_{1}.png ".format(save_dir,
                                                                                                             week,
                                                                                                   ))

                if clf_name in ['DecisionTreeRegressor', 'RandomForestRegressor', "GradientBoostingRegressor"]:
                    df = pd.DataFrame(collections.OrderedDict({"variable": trn_data[week].columns[3:],
                                                               "feature importance" : best_model.feature_importances_}))

                    df.sort_values("feature importance", ascending=False, inplace=True)
                    df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)


                with open(os.path.join(save_dir, "specifications.txt"), "w") as f:
                    f.write("best model {}\n".format(repr(best_model)))
                    if nationality == "전체":
                        f.write("train shape {}\n".format(trn_data[week].loc[
                                                              (trn_data[week]["target"] > cond)].shape))
                        f.write("test shape {}\n".format(test_data[week].loc[(test_data[week]["target"] > cond)
                                                                             ].shape))
                        f.write("validation shape {}\n".format(val_data[week].loc[(val_data[week]["target"] > cond)
                                                                                  ].shape))
                    else:
                        f.write("train shape {}\n".format(trn_data[week].loc[
                                                              (trn_data[week]["target"] > cond) & (
                                                              trn_data[week]["국적"]
                                                              == nationality)].shape))
                        f.write("test shape {}\n".format(test_data[week].loc[(test_data[week]["target"] > cond) &
                                                                             (test_data[week]["국적"]
                                                                              == nationality)].shape))

                        f.write("validation shape {}\n".format(val_data[week].loc[(val_data[week]["target"] > cond) &
                                                                                  (val_data[week]["국적"]
                                                                                   == nationality)].shape))

                    f.write("validation score : {}\n\n".format(val_score.iloc[0]))
                    f.write("test score : {}\n".format(test_score.loc[val_score.index[0]]))
                    f.write("\n\n\n\n\n")
                    f.write("한국 over 300000 validation: {}\n".format(evaluate(best_model, val_data[week], False,
                                                                               "한국", 300000)[1]))

                    f.write("미국 over 300000 mse validation: {}\n".format(evaluate(best_model, val_data[week], False,
                                                                               "미국", 300000)[1]))

                    f.write("전체 over 300000 mse validation: {}\n".format(evaluate(best_model, val_data[week], False,
                                                                                  "전체", 300000)[1]))

                    f.write("한국 over 300000 test: {}\n".format(evaluate(best_model, test_data[week], False,
                                                                              "한국", 300000)[1]))

                    f.write("미국 over 300000 mse test: {}\n".format(evaluate(best_model, test_data[week], False,
                                                                                  "미국", 300000)[1]))

                    f.write("전체 over 300000 mse test: {}\n".format(evaluate(best_model, test_data[week], False,
                                                                                  "전체", 300000)[1]))



                    f.write("\n\n\n")
                    f.write(str(list(trn_data[week].columns)))



            model_score.to_csv(os.path.join(self.result_dir, clf_name, "weekly_summary.csv"))

        trn_data = self.data["train"]
        val_data = self.data["validation"]
        test_data = self.data["test"]
        train(clf)