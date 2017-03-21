'''
Created on 2015. 5. 15.
@author: jinwon
'''
'''
주말 관객수 csv와 영화 제목 맞추기
'''
import pandas as pd
import os
import string
import re
import numpy as np
from datetime import datetime, timedelta

class KobisFunMatcher():

    def __init__(self, result_dir=None):
        if result_dir:
            self.result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result', str(result_dir))
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
        else:
            self.result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result')


    def find_failures_search_possbile_match(self, data, fun):
        print("start finding matches")

        # data = pd.read_csv(os.path.join(result_dir, "daily_seats",
        #                                 'target_data_SAT_from_2010-01-01_to_2016-09-04_top_10.csv'))
        # fun = pd.read_csv(os.path.join(result_dir, 'funlife_horizontal_111-461.csv'))

        fun_name = fun.loc[:, '영화 제목']
        data_name = data.loc[:, '영화명']

        ## get name
        fun_name = [re.sub("[^가-힣\w\d]", "", temp_id) for temp_id in fun_name]
        fun_name = [str.lower(x) for x in fun_name]

        data_name = [re.sub("[^가-힣\w\d]", "", temp_id) for temp_id in data_name]
        data_name = [str.lower(x) for x in data_name]

        ## get date
        fun_date = [re.search("\d{4}-\d{2}-\d{2}", x ).group() for x in fun["Identifier"]]
        fun_date = [datetime.strptime(x, "%Y-%m-%d")for x in fun_date]
        fun_date = pd.Series(fun_date)

        data_date = [re.search("\d{4}-\d{2}-\d{2}", x ).group() for x in data["Identifier"]]
        data_date = [datetime.strptime(x, "%Y-%m-%d")for x in data_date]
        data_date = pd.Series(data_date)


        ### find match
        success_data_to_fun_idx = {}
        failure_data_name = []
        failure_count = 0

        ## 먼저 제목이 같은지 체크하고,
        ## 제목이 같다면 개봉일이 비슷한지 체크하고(차이나는 경우가 있음)
        ## 여러개가 매칭된다면, 일일히 체크하고,
        ## 매칭이 하나도 안된다면 이도 일일히 체크한다. (possbile_match.txt를 보고 possible_match_found.txt를 만든다.)

        for idx, id in enumerate(data_name):
            match_id = np.where([re.match("^" + id + "$", x) for x in fun_name])[0]

            if len(match_id) == 0:
                failure_data_name.append(id)
                failure_count += 1
            elif len(match_id) == 1:
                if abs(data_date[idx] - fun_date[match_id[0]]) < timedelta(7):
                    success_data_to_fun_idx[idx] = match_id[0]
                    continue
            else:
                true_count = 0
                true_id = None
                for i in match_id:
                    if abs(data_date[idx] - fun_date[i]) < timedelta(7):
                        true_count += 1
                        if true_count == 1:
                            true_id = i
                        else:
                            print("multiple matches")
                            print(id)
                            print([[fun_name[i] for i in match_id]])
                            failure_count += 1
                            break
                if true_count == 1:
                    success_data_to_fun_idx[idx] = true_id

        print("failure count :", failure_count)


        ### finding candidates for movies that are not matched
        not_found = list(set(range(len(fun_name))) - set(success_data_to_fun_idx.values()))

        fun_name_not_found = [fun_name[i] for i in not_found]

        possible_count = 0

        """ 이 possible match 부분은 수동으로 possible_match.txt 파일을 보면서 matching을 해야한다.
        그리고 이 matching된 파일을 possible_match_found.txt로 만든다.

        """

        f = open(os.path.join(self.result_dir, "possbile_match.txt"), "w")
        for x in failure_data_name:
            for j in fun_name_not_found:
                sim = sum([char in str(j) for char in str(x)]) / len(x)
                length_diff = len(j) - len(x)
                if sim > 0.5 and length_diff < 5:
                    possible_count += 1
                    f.write("possibile match\n")
                    f.write(x+"\n")
                    f.write(j+"\n\n")

        f.close()

        print("possbile count : ", possible_count)

        self.data = data
        self.fun = fun
        self.data_name = data_name
        self.data_date = data_date
        self.fun_name = fun_name
        self.fun_date = fun_date
        self.success_data_to_fun_idx = success_data_to_fun_idx


    def add_matches_found(self, file_name = "possible_match_found.txt"):
        print("adding matches manually found")
        #### make matches according to possbile_match_found

        f = open(os.path.join(self.result_dir, file_name))

        line = f.readline()
        while(line !=""):
            line = f.readline()
            if line.startswith("possibile match"):
                data_name_match = f.readline().rstrip()
                fun_name_match = f.readline().rstrip()
                data_idx = self.data_name.index(data_name_match)
                fun_idx = self.fun_name.index(fun_name_match)
                if data_idx in self.success_data_to_fun_idx.keys():
                    print("already in error")
                print("added: ", data_name_match, fun_name_match)
                self.success_data_to_fun_idx[data_idx] = fun_idx


    def generate_training_data(self, target_col, additional_cols=None):
        print("Join kobis data and fun life data")
        filtered_fun = self.fun.loc[list(self.success_data_to_fun_idx.values())]

        for data_idx, fun_idx in self.success_data_to_fun_idx.items():
            filtered_fun.loc[fun_idx, 'Identifier'] = self.data.loc[data_idx, 'Identifier']
            filtered_fun.loc[fun_idx, "target"] = self.data.loc[data_idx, target_col]
            filtered_fun.loc[fun_idx, "국적"] = self.data.loc[data_idx, "국적"]
            filtered_fun.loc[fun_idx, "장르"] = self.data.loc[data_idx, "장르"]
            filtered_fun.loc[fun_idx, "등급"] = self.data.loc[data_idx, "등급"]


        filtered_fun['국적'] = filtered_fun.apply(lambda row: row['국적'] if row['국적'] in ['한국', '미국']  else
        '기타', axis=1)
        genre = list()
        for i in filtered_fun["장르"]:
            genre.extend(i.split(","))
        genre = list(np.unique(genre))

        grade = list(np.unique(filtered_fun["등급"]))

        onehot_genre = np.zeros((len(filtered_fun), len(genre)))

        for idx, row in enumerate(filtered_fun["장르"]):
            items = row.split(",")
            for item in items:
                onehot_genre[idx, genre.index(item)] = 1

        onehot_grade = np.zeros((len(filtered_fun), len(grade)))

        for idx, row in enumerate(filtered_fun["등급"]):
            onehot_grade[idx, grade.index(row)] = 1

        ### 상대 절대만 넣을 경우
        cols_to_include = ["Identifier", "target", "국적"]
        cols_to_include.extend([x for x in filtered_fun.columns if "상대" in x or "절대" in x])

        if additional_cols:
            assert isinstance(additional_cols, "list")
            cols_to_include = cols_to_include + additional_cols

        training_data = filtered_fun.loc[:,  cols_to_include]

        training_data.reset_index(drop=True, inplace=True)
        training_data = pd.concat([training_data, pd.DataFrame(onehot_genre, columns=genre)], axis=1)
        training_data = pd.concat([training_data, pd.DataFrame(onehot_grade, columns=grade)], axis=1)

        # training_data.sort_values("target_SAT", ascending=False, inplace=True)

        training_data.to_csv(os.path.join(self.result_dir, "training_data_{}.csv".format(target_col)), index=False)
        return training_data


def run_name_matcher(kobis_data, funlife_data, target_col, year_start):
    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result', str(year_start))
    training_data_dir = os.path.join(result_dir, "training_data_{}.csv".format(target_col))
    if os.path.exists(training_data_dir):
        print("returning existing training data")
        training_data = pd.read_csv(training_data_dir)
    else:
        matcher = KobisFunMatcher(year_start)
        # result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result')
        # data = pd.read_csv(os.path.join(result_dir, "daily_seats",
        #                                 'target_data_SAT_from_2010-01-01_to_2016-09-04_top_10.csv'))
        # fun = pd.read_csv(os.path.join(result_dir, 'funlife_horizontal_111-461.csv'))

        matcher.find_failures_search_possbile_match(kobis_data, funlife_data)
        print("check the possible_match.txt and find real matches in the file possible_match_found.txt")
        response = input("after making the match type continue:")
        if response == "continue":
            return
        training_data = matcher.generate_training_data(target_col)

    return training_data

