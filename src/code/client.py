from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.code.classification import Classification
from src.code.funlife_original import run_funlife
from src.code.kobis_scrape_preprocess import kobis_run
from src.code.name_match_revised import run_name_matcher
import collections
from datetime import datetime
import re
import pandas as pd

def run(start_date, end_date, top_k, weekday, fun_start, fun_end, clf_list, param_grid_list, nationality,
        data_restriction, target_col):
    start_year = start_date.split('-')[0]
    kobis_data = kobis_run(start_date= start_date, end_date= end_date, top_k= top_k,
                           weekday= weekday, seats_or_aud='aud')
    funlife_data = run_funlife(111, 461)
    training_data = run_name_matcher(kobis_data, funlife_data, target_col, start_year)

    ### preprocess

    Date_time = [re.search("\d{4}-\d{2}-\d{2}", x).group() for x in training_data["Identifier"]]
    Date_time = [datetime.strptime(x, "%Y-%m-%d") >= datetime.strptime(start_date, "%Y-%m-%d") for x in Date_time]

    training_data = training_data.iloc[Date_time]

    ## fill in missing value
    # training_data = training_data.fillna(training_data.mean())
    # assert sum(training_data.isnull().any(axis=1)) == 0, "not 0 missing exists"

    ### over 70 thousand
    training_data.loc[training_data["target"] > 700000, "target"] = 700000

    # model = Classification('ordinary')
    model_specification = "over_{}".format(data_restriction)
    model = Classification(model_specification+"_"+nationality+"_"+target_col+"_"+start_year)
    # training_data = training_data.loc[training_data["target"] > data_restriction * 10000]

    training_data.reset_index(drop = True, inplace=True)
    model.read_data(training_data, start_year)

    for clf, param_grid in zip(clf_list, param_grid_list):
        model.train_classifier(clf=clf, param_grid=param_grid, eval_metric="mse", nationality=nationality,
                               cond = data_restriction * 10000)

def run_process():
    clf_dt = DecisionTreeRegressor()
    param_grid_dt = collections.OrderedDict({'max_leaf_nodes' : [6, 8, 10],
                                             'min_samples_leaf' : [3, 5, 10, 20]
                                             })
    clf_list = [clf_dt]
    param_grid_list = [param_grid_dt]

    #
    # clf_svm = SVR()
    # param_grid_svm = collections.OrderedDict({'C': [1, 10, 100, 1000], 'kernel': ['rbf', 'linear'],
    #                                           'gamma' : ['auto', 0.001, 0.01, 0.1],
    #                                           'epsilon' : [0.001, 0.005, 0.01]})
    #
    # clf_rf = RandomForestRegressor()
    # param_grid_rf = collections.OrderedDict({"n_estimators": [100, 300],
    #                                          "max_depth": [3, 4, 5],
    #                                          "min_samples_split": [3, 5, 7]})
    #
    # clf_gb = GradientBoostingRegressor()
    # param_grid_gb = collections.OrderedDict({"n_estimators": [100, 300],
    #                  "subsample": [0.8], "min_samples_split": [5, 10],
    #                  "max_depth": [3, 4, 5]})
    #
    # clf_knn = KNeighborsRegressor()
    # param_grid_knn = {"n_neighbors": [5, 10, 15], "weights": ["uniform", "distance"]}



    # clf_list = [clf_svm, clf_knn]
    # param_grid_list = [param_grid_svm, param_grid_knn]
    print('2012 training')
    for type in ['전체', '한국', '미국']:
        for cond in [0, 10]:
            run("2012-01-01", "2016-09-04", 10, "SAT", 111, 461, clf_list, param_grid_list, type, cond, "관객수")


def merge_data(merge_file_list):
    data = pd.concat(merge_file_list)



if __name__ =="__main__":
    # funlife_data = run_funlife(1, 110)
    # funlife_data = run_funlife(461, 489)

    # kobis_data = kobis_run(start_date='2004-01-01', end_date='2008-12-31',
    #                        top_k=10,
    #                        weekday='SAT', seats_or_aud='aud')

    # for i in range(20):
    #     try:
    #         kobis_data = kobis_run(start_date='2004-01-01', end_date='2008-12-31',
    #                                top_k=10,
    #                                weekday='SAT')
    #     except Exception as e:
    #         print(str(e))
    #         continue
    #     break
    #
    #

    for i in range(20):
        try:
            kobis_data = kobis_run(start_date='2016-09-05', end_date='2017-03-18',
                                   top_k=10,
                                   weekday='SAT', seats_or_aud='aud')
        except Exception as e:
            print(str(e))
            continue
        break





