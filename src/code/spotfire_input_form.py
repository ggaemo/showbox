import re
import os
import pandas as pd
from datetime import timedelta
from datetime import datetime
import numpy as np
from scipy.stats import kendalltau, spearmanr
import requests
import json
import shutil

def add_holiday_info(df):
    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result')

    df['날짜'] = pd.to_datetime(df['날짜'])
    df['개봉일'] = pd.to_datetime(df['개봉일'])

    headers = {'TDCProjectKey': 'be30d25d-65a4-457f-8790-881d0f75852c'}
    holiday_list = list()
    for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016]:
        ### 2014년 이후 부터는 skt api이용
        page = requests.get("https://apis.sktelecom.com/v1/eventday/days?type=h,i&year={}".format(year),
                            headers=headers)
        newDictionary = json.loads(str(page.content, encoding='utf-8'))

        for date in newDictionary['results']:
            to_date = (int(date['year']), int(date['month']), int(date['day']))
            holiday_list.append((datetime(*to_date), date['name']))

    for holiday in holiday_list:
        date, name = holiday
        df.loc[df['개봉일'] == date, '공휴일여부(개봉일)'] = name
        df.loc[df['날짜'] == date, '공휴일여부'] = '공휴일'
        df.loc[df['날짜'] == date, '공휴일'] = name
        df.loc[df['날짜'] == date, '휴일여부'] = '휴일'


    # 2013년 까지는 이전에 수집한 자료 이용
    prev_data = pd.read_excel(os.path.join(result_dir, '자료01_01_기본정보_입력데이터.xlsx'), sheetname='영화별정보')

    prev_data = prev_data[['개봉일', '공휴일여부(개봉일)']].dropna()

    df['공휴일여부'] = 'X'
    df['휴일여부'] = '평일'
    for i in prev_data.index:
        date = prev_data.loc[i, '개봉일']
        name = prev_data.loc[i, '공휴일여부(개봉일)']
        df.loc[df['개봉일'] == date, '공휴일여부(개봉일)'] = name
        df.loc[df['날짜'] == date, '공휴일여부'] = '공휴일'
        df.loc[df['날짜'] == date, '공휴일'] = name
        df.loc[df['날짜'] == date, '휴일여부'] = '휴일'

    df['휴일여부'] = df.apply(lambda row: '휴일' if row['날짜'].weekday() >= 5 else '평일', axis=1)

    return df


def merge_kobis_data(start, end, top_k):
    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result')

    if os.path.exists(
            os.path.join(result_dir, 'daily_seats', "daily_aud_seats_merged_from_{}_to_{}.csv".format(start, end))):
        print('get existing data')
        tmp = pd.read_csv(os.path.join(result_dir, 'daily_seats', "daily_aud_seats_merged_from_{}_to_{}.csv".format(
            start, end)))
        tmp['날짜'] = pd.to_datetime(tmp['날짜'])
        tmp['개봉일'] = pd.to_datetime(tmp['개봉일'])
        return tmp
    print('merging data')

    aud_file = 'daily_box_all_movies_from_{}_to_{}_top_{}_run_merged.csv'.format(start, end, top_k)
    seats_file = 'daily_seats_all_movies_from_{}_to_{}_top_{}.csv'.format(start, end, top_k)

    aud_data = pd.read_csv(os.path.join(result_dir, 'daily_seats', aud_file))
    seats_data = pd.read_csv(os.path.join(result_dir,  'daily_seats', seats_file), thousands=',')

    seats_data.drop("순위", axis=1, inplace=True)  # 중복됨 그리고 이건 좌점율 순위임

    after_2010 = [bool(re.match("^201", str(x))) for x in aud_data["개봉일"]]
    aud_data = aud_data.loc[after_2010]
    after_2010 = [bool(re.match("^201", str(x))) for x in seats_data["개봉일"]]
    seats_data = seats_data.loc[after_2010]

    aud_data["개봉일"] = pd.to_datetime(aud_data["개봉일"], format="%Y-%m-%d")
    aud_data["날짜"] = pd.to_datetime(aud_data["날짜"], format="%Y-%m-%d")
    seats_data["개봉일"] = pd.to_datetime(seats_data["개봉일"], format="%Y-%m-%d")
    seats_data["날짜"] = pd.to_datetime(seats_data["날짜"], format="%Y-%m-%d")

    aud_data["Identifier"] = aud_data.apply(lambda row: "{}_{}".format(row["영화명"], row["개봉일"].date()),
                                            axis=1)

    seats_data["Identifier"] = seats_data.apply(lambda row: "{}_{}".format(row["영화명"], row["개봉일"].date()),
                                                axis=1)

    cols_to_use = list(seats_data.columns.difference(aud_data.columns))
    cols_to_use.extend(["Identifier", "날짜"])

    data = pd.merge(aud_data, seats_data[cols_to_use], on=["Identifier", "날짜"], how="left")
    data["개봉일차"] = (data["날짜"] - data["개봉일"])
    data["개봉일차"] = [x.days + 1 for x in data["개봉일차"]]

    data["개봉주차"] = data["개봉일차"] // 7 + 1

    data["주말여부"] = ["주말" if x.weekday() in [5, 6] else "주중" for x in data["날짜"]]

    agg_data = data.groupby("날짜").aggregate(np.sum)

    data["상영횟수/스크린수"] = data["상영횟수"] / data["스크린수"]
    data["좌석수/상영횟수"] = data["좌석수"] / data["상영횟수"]

    data["총관객수"] = [agg_data.loc[x, "관객수"] for x in data["날짜"]]
    data["총좌석수"] = [agg_data.loc[x, "좌석수"] for x in data["날짜"]]
    data["총상영횟수"] = [agg_data.loc[x, "상영횟수"] for x in data["날짜"]]
    data["총스크린수"] = [agg_data.loc[x, "스크린수"] for x in data["날짜"]]

    data["관객MS"] = data["관객수"] / data["총관객수"]
    data["좌석MS"] = data["좌석수"] / data["총좌석수"]
    # data["관객MS"] = data["관객MS"].apply(lambda row: "{0:.2f} %".format(row))
    # data["좌석MS"] = data["좌석MS"].apply(lambda row: "{0:.2f} %".format(row))

    data['좌점율'] = data['관객수'] / data['좌석수']
    # data['좌점율'] = data['좌점율'].apply(lambda row: '{0:.2f} %'.format(row))

    data['국적'] = data.apply(lambda row: row['대표국적'] if row['대표국적'] in ['한국', '미국'] else '기타', axis=1)


    data.to_csv(os.path.join(result_dir, 'daily_seats',"daily_aud_seats_merged_from_{}_to_{}.csv".format(start, end)),
                index=False)
    return data


def spotfire_1(start, end, top_k, year_start):
    print('spotfire 1')
    result_dir = '/home/jinwon/PycharmProjects/showbox/src/result/'

    df = pd.DataFrame(columns = ['Identifier','영화명', '국적', '배급사', '등급', '감독', '배우', '배우(3순위)', '장르', '상영시간',
                                 '개봉일', '요일(개봉일)', '관객수(최종)', '관객수(개봉주)', '관객수기여도(개봉주)',
                                 '관객수(개봉일)', '순위(개봉일)', '좌점율(개봉일)', '스크린당상영횟수(개봉일)', '회당좌석수(개봉일)',
                                     '좌석MS(개봉일)', '관객MS(개봉일)', '박스유도율', '관객수(첫토)',
                                     '순위(첫토)', '좌점율(첫토)', '스크린당상영횟수(첫토)', '회당좌석수(첫토)',
                                     '좌석MS(첫토)', '관객MS(첫토)', '좌석드랍률', '관객드랍률', '공휴일여부(개봉일)'
                                     ])

    data = merge_kobis_data(start, end, top_k)
    data = add_holiday_info(data)

    ### training에 쓰인 데이터만 넣는다

    start_year = start.split('-')[0]
    training_data = pd.read_csv(os.path.join(result_dir, start_year,
                                             'training_data_관객수_imputation_{}.csv'.format(start_year)))
    Date_time = [re.search("\d{4}-\d{2}-\d{2}", x).group() for x in training_data["Identifier"]]
    Date_time = [datetime.strptime(x, "%Y-%m-%d") >= datetime(year_start, 1, 1) for x in Date_time]
    training_data = training_data.iloc[Date_time]

    data.sort_values(['Identifier', '날짜'])

    existing_columns = ['Identifier','영화명', '배급사', '등급', '감독', '배우', '장르', '개봉일', '상영시간',
                        '공휴일여부(개봉일)']

    weekday_dic = {0 : "월",
                   1 : "화",
                   2 :"수",
                   3: "목",
                   4: "금",
                   5: "토" ,
                   6: "일"}

    def filter(column, condition):
        filtered = data.loc[data[column] == condition]
        filtered.reset_index(drop=True, inplace=True)
        return filtered

    for idx, movie in enumerate(training_data['Identifier'].unique()):
        movie_data = filter('Identifier', movie)
        #     print(movie)
        #     print(len(movie))
        #     print(movie_data.iloc[:5])
        movie_data.sort_values('날짜')
        open_date = movie_data.loc[0, '개봉일']

        sat_date = open_date + timedelta(days=5 - open_date.weekday())

        mon_date = open_date + timedelta(days=7 - open_date.weekday())
        sun_date = open_date + timedelta(days=6 - open_date.weekday())

        try:
            df.loc[idx, '배우(3순위)'] = ', '.join(movie_data.loc[0, '배우'].split(',')[:3])
        except AttributeError:  # 없을때
            df.loc[idx, '배우(3순위)'] = np.NaN
        df.loc[idx, '요일(개봉일)'] = weekday_dic[movie_data.loc[0, '개봉일'].weekday()]
        df.loc[idx, '관객수(최종)'] = movie_data.loc[len(movie_data) - 1, '누적관객수']
        try:
            df.loc[idx, '관객수(개봉주)'] = movie_data.loc[movie_data['날짜'] == open_date + timedelta(days=6), '누적관객수'].values[
                0]
        except IndexError:  # 개봉주 데이터가 없으르때 (아직 수집이 안되서)
            df.loc[idx, '관객수(개봉주)'] = np.NaN
        df.loc[idx, '관객수기여도(개봉주)'] = df.loc[idx, '관객수(개봉주)'] / df.loc[idx, '관객수(최종)']

        df.loc[idx, '관객수(개봉일)'] = movie_data.loc[movie_data['날짜'] == open_date, '관객수'].values[0]
        df.loc[idx, '순위(개봉일)'] = movie_data.loc[movie_data['날짜'] == open_date, '순위'].values[0]
        df.loc[idx, '좌점율(개봉일)'] = movie_data.loc[movie_data['날짜'] == open_date, '좌점율'].values[0]
        df.loc[idx, '스크린당상영횟수(개봉일)'] = movie_data.loc[movie_data['날짜'] == open_date, "상영횟수/스크린수"].values[0]
        df.loc[idx, '회당좌석수(개봉일)'] = movie_data.loc[movie_data['날짜'] == open_date, "좌석수/상영횟수"].values[0]
        df.loc[idx, '좌석MS(개봉일)'] = movie_data.loc[movie_data['날짜'] == open_date, "좌석MS"].values[0]
        df.loc[idx, '관객MS(개봉일)'] = movie_data.loc[movie_data['날짜'] == open_date, "관객MS"].values[0]

        df.loc[idx, '관객수(첫토)'] = movie_data.loc[movie_data['날짜'] == sat_date, '관객수'].values[0]
        df.loc[idx, '순위(첫토)'] = movie_data.loc[movie_data['날짜'] == sat_date, '순위'].values[0]
        df.loc[idx, '좌점율(첫토)'] = movie_data.loc[movie_data['날짜'] == sat_date, '좌점율'].values[0]
        df.loc[idx, '스크린당상영횟수(첫토)'] = movie_data.loc[movie_data['날짜'] == sat_date, "상영횟수/스크린수"].values[0]
        df.loc[idx, '회당좌석수(첫토)'] = movie_data.loc[movie_data['날짜'] == sat_date, "좌석수/상영횟수"].values[0]
        df.loc[idx, '좌석MS(첫토)'] = movie_data.loc[movie_data['날짜'] == sat_date, "좌석MS"].values[0]
        df.loc[idx, '관객MS(첫토)'] = movie_data.loc[movie_data['날짜'] == sat_date, "관객MS"].values[0]



        sun_aud = movie_data.loc[movie_data['날짜'] == sun_date, '관객수'].values[0]
        try:
            mon_aud = movie_data.loc[movie_data['날짜'] == mon_date, '관객수'].values[0]
        except IndexError: # not existing
            mon_aud = 0

        df.loc[idx, '관객드랍률'] = (mon_aud - sun_aud) / sun_aud

        sun_seats = movie_data.loc[movie_data['날짜'] == sun_date, '좌석수'].values[0]
        try:
            mon_seats = movie_data.loc[movie_data['날짜'] == mon_date, '좌석수'].values[0]
        except IndexError: # not existing
            mon_seats = 0

        df.loc[idx, '좌석드랍률'] = (mon_seats-sun_seats) / sun_seats


        df.loc[idx, '박스유도율'] =  data.loc[data['날짜'] == open_date, '총관객수'].iloc[0] \
                                / data.loc[data['날짜'] == open_date + timedelta(days=-1), '총관객수'].iloc[0]

        # nationality = movie_data.loc[0, '대표국적']
        # if nationality not in ['한국', '미국']:
        #     nationality = '기타'
        # df.loc[idx, '국적'] = nationality
        df.loc[idx, '국적'] = movie_data.loc[0, '국적']
        df.loc[idx, existing_columns] = movie_data.loc[0, existing_columns]

    # df['관객수기여도(개봉주)'] = df['관객수기여도(개봉주)'].apply(lambda row: "{0:.2f} %".format(row))
    # df['좌석드랍률'] = df['좌석드랍률'].apply(lambda row: "{0:.2f} %".format(row))

    #### 인지 선호 추가
    fun_var = [re.match('.*_-\d+', x).group() for x in training_data.columns if re.match('(.*)_-\d+', x)]
    df = df.merge(training_data.loc[:,['Identifier'] + fun_var], how = 'left', on='Identifier')

    fun_var_type = np.unique([re.match('(.*)_-\d+', x).group(1) for x in training_data.columns if re.match('(.*)_-\d+',
                                                                                                         x)])

    df_2 = pd.DataFrame(columns=['영화제목', '조사기관', '종류', '6주전', '5주전',
                               '4주전', '3주전', '2주전', '1주전'])

    for movie_idx, movie in enumerate(training_data["Identifier"]):
        tmp = pd.DataFrame(columns=['영화제목', '조사기관', '종류', '6주전', '5주전',
                                    '4주전', '3주전', '2주전', '1주전'])

        for idx, i in enumerate(fun_var_type):
            tmp.loc[idx, '종류'] = i
            for j in [1, 2, 3, 4, 5, 6]:
                a = training_data.iloc[movie_idx]["{}_-{}".format(i, j)]
                tmp.loc[idx, "{}주전".format(j)] = a

        tmp['영화제목'] = movie
        tmp['조사기관'] = '즐거운인생'
        df_2 = df_2.append(tmp, ignore_index=True)

    writer = pd.ExcelWriter(os.path.join(result_dir, 'spotfire', '자료01_01_기본정보_입력데이터.xlsx'))
    if not os.path.exists(os.path.join(result_dir, 'spotfire')):
        os.makedirs(os.path.join(result_dir, 'spotfire'))

    df.to_excel(writer, sheet_name='영화별정보', index=False)
    df_2.to_excel(writer, sheet_name='인지선호', index=False)
    writer.save()


def spotfire_2(start, end, top_k):
    print('spotfire 2')
    result_dir = '/home/jinwon/PycharmProjects/showbox/src/result/'

    df = pd.DataFrame(columns = ['영화명', '개봉일', '관객수', '누적관객수', '좌석수', '스크린수', '상영횟수',
                                     '날짜', '개봉일차', '개봉주차', '순위(관객수)', '상영시간',
                                     '국적', '등급',  "주말여부",'총관객수', '총스크린수', '총좌석수', '상영횟수/스크린수',
                                 '좌석수/상영횟수', '좌석MS', '관객MS', '좌점율', '휴일여부'
                                     ])

    data = merge_kobis_data(start, end, top_k)
    data = add_holiday_info(data)
    # data['국적'] = data.apply(lambda row: rowㅁ['대표국적'] if row['대표국적'] in ['한국', '미국'] else '기타', axis = 1)

    cols_except_rank = list(df.columns)
    cols_except_rank.remove('순위(관객수)')
    df[cols_except_rank] = data[cols_except_rank]
    df['순위(관객수)'] = data['순위']
    df = df.loc[df['개봉일차'] >= 0]
    df.to_excel(os.path.join(result_dir, 'spotfire', '자료01_02_트렌드분합석_입력데이터.xlsx'),
                sheet_name='통합', index=False)


def spotfire_3(start, end, top_k):
    print('spotfire 3')
    result_dir = '/home/jinwon/PycharmProjects/showbox/src/result/'
    df = pd.DataFrame(columns=['영화명', '개봉일', '누적관객수', '요일(시작 마지막)', '개봉월일'
                               ])

    data = pd.read_excel(os.path.join(result_dir, 'spotfire','자료01_01_기본정보_입력데이터.xlsx'),
                sheet_name='영화별정보')

    weekday_dic = {0 : "월",
                   1 : "화",
                   2 :"수",
                   3: "목",
                   4: "금",
                   5: "토" ,
                   6: "일"}

    df[['영화명', '개봉일', '누적관객수']] = data[['영화명', '개봉일', '관객수(개봉주)']]
    df['개봉일'] = pd.to_datetime(df['개봉일'])
    df['요일(시작 마지막)'] = data.apply(lambda row: '{}_{}'.format(weekday_dic[row['개봉일'].weekday()], weekday_dic[row[
        '개봉일'].weekday() + 1]), axis=1)


    df = df.loc[df['누적관객수'] > 1000000]
    df.to_excel(os.path.join(result_dir, 'spotfire', '자료01_03_개봉후7일누적관객수.xlsx'),
                sheet_name='개봉후7일관객수백만이상', index=False)


def spotfire_4(start, end, top_k, result_fnames):
    print('spotfire 4')
    '''
    자료02_01_예측모델_입력데이터.xlsx
    :param result_fnames:
    :return:
    '''
    start_year = start.split('-')[0]
    result_dir = '/home/jinwon/PycharmProjects/showbox/src/result/'
    questionnaire_root_file = os.path.join(result_dir, start_year,
                                           'training_data_관객수_imputation_{}.csv'.format(start_year))
    questionnaire = pd.read_csv(questionnaire_root_file)
    cols_to_use = [re.search('.*_-\d', x).group() for x in questionnaire.columns if re.search('.*_-\d', x)]
    cols_to_use.append('Identifier')
    questionnaire = questionnaire[cols_to_use]
    # questionnaire.drop('target', axis=1, inplace=True)

    movie_info = pd.read_excel(os.path.join( result_dir,'spotfire' ,'자료01_01_기본정보_입력데이터.xlsx'),
                               sheetname='영화별정보')

    chosen_columns = ['Identifier','영화명', '등급', '개봉일', '관객수(첫토)', '국적']
    movie_info = movie_info[chosen_columns]

    spotfire_input_original = movie_info.merge(questionnaire, on='Identifier', how='inner')

    for result_fname in result_fnames:
        your_concept_root_folder = r'/home/jinwon/PycharmProjects/showbox/src/result/%s/DecisionTreeRegressor/' % (
            result_fname)
        spotfire_input =spotfire_input_original.copy()
        if not os.path.exists(os.path.join(result_dir, 'spotfire', 'model',result_fname)):
            os.makedirs(os.path.join(result_dir, 'spotfire', 'model', result_fname))
        for path_name_1, folders, _ in os.walk(your_concept_root_folder):

            # 1,2,3,4,5,6
            for folder_name in folders:
                for path_name_2, _, fnames, in os.walk(os.path.join(path_name_1, folder_name)):

                    pred = pd.DataFrame(columns=['Identifier', 'target', '국적', 'prediction', 'group'])

                    for fname in fnames:

                        # 모든 영화 데이터에 group label이 달린 DataFrame 생성

                        # Append
                        # decision_tree_png = shutil.copyfile(os.path.join(path_name_2, 'decision_tree_{}'.format()), )
                        if fname in ["train_pred.csv", "validation_pred.csv", "test_pred.csv"]:  # test / train /
                            # validation
                            tmp_pred = pd.read_csv(os.path.join(path_name_2, fname))
                            pred = pred.append(tmp_pred, ignore_index=True)

                        if re.match('decision_tree_', fname):
                            shutil.copyfile(os.path.join(path_name_2, fname), os.path.join(result_dir, 'spotfire',
                                                                                           'model',result_fname, fname))
                    pred.drop(['target','국적'], axis=1, inplace=True)
                    pred.columns = ['Identifier', 'prediction{}주전'.format(folder_name), '그룹명(관객수{}주전)'.format(folder_name)]

                    # cols_to_use = list(pred.columns.difference(spotfire_input.columns))
                    # cols_to_use.append("Identifier")
                    spotfire_input = spotfire_input.merge(pred, on="Identifier", how='left')


        spotfire_input.loc[spotfire_input["Identifier"].str.contains('2010'), 'data_type'] = 'train'
        spotfire_input.loc[spotfire_input["Identifier"].str.contains('2011'), 'data_type'] = 'train'
        spotfire_input.loc[spotfire_input["Identifier"].str.contains('2012'), 'data_type'] = 'train'
        spotfire_input.loc[spotfire_input["Identifier"].str.contains('2013'), 'data_type'] = 'train'
        spotfire_input.loc[spotfire_input["Identifier"].str.contains('2014'), 'data_type'] = 'train'
        spotfire_input.loc[spotfire_input["Identifier"].str.contains('2015'), 'data_type'] = 'validation'
        spotfire_input.loc[spotfire_input["Identifier"].str.contains('2016'), 'data_type'] = 'test'

        # spotfire_input.drop('Identifier', axis=1, inplace=True)
        spotfire_input = spotfire_input[spotfire_input['그룹명(관객수1주전)'].isnull() == False]

        # spotfire_input.to_csv(your_concept_root_folder+'spotfire_input.csv', index = False)

        spotfire_input.to_excel(os.path.join(result_dir, 'spotfire', 'model',result_fname ,'자료02_01_예측모델_입력데이터.xlsx'),
                    sheet_name='통합', index=False)


def spotfire_5():
    print('spotfire 5')
    result_dir = '/home/jinwon/PycharmProjects/showbox/src/result/'
    data = pd.read_excel(os.path.join(result_dir, 'spotfire','자료01_01_기본정보_입력데이터.xlsx'),
                sheet_name='영화별정보')

    cols = [re.search('.*_-\d', x).group() for x in data.columns if re.search('.*_-\d', x)]

    target_cols = ['관객수(최종)', '관객수(개봉일)', '관객수(첫토)', '좌석MS(개봉일)', '좌석MS(첫토)', '좌점율(개봉일)', '좌점율(첫토)']

    corr = pd.DataFrame(columns =target_cols, index=cols+target_cols)
    for i in cols + target_cols:
        for j in target_cols:
            corr.loc[i, j] = kendalltau(data[i], data[j], nan_policy='omit')[0]
            # corr.loc[i, j] = spearmanr(data[i], data[j], nan_policy='omit')[0]

    corr.to_excel(os.path.join(result_dir, 'spotfire', '자료02_02_상관관계분석.xlsx'))



def summarize_results(start_year):
    result_root_folder = os.path.join(os.path.dirname(os.getcwd()), 'result')

    patterns = ["한국 over 300000 validation: OrderedDict\(\[\('mse', (\d+\.\d+)\)",
                "미국 over 300000 mse validation: OrderedDict\(\[\('mse', (\d+\.\d+)\)",
                "전체 over 300000 mse validation: OrderedDict\(\[\('mse', (\d+\.\d+)\)",
                "한국 over 300000 test: OrderedDict\(\[\('mse', (\d+\.\d+)\)",
                "미국 over 300000 mse test: OrderedDict\(\[\('mse', (\d+\.\d+)\)",
                "전체 over 300000 mse test: OrderedDict\(\[\('mse', (\d+\.\d+)\)"]

    result_summary = pd.DataFrame(columns=['Training data', 'Model', 'Week', 'over30_val_한국',
                                           'over30_val_미국', 'over30_val_전체', 'over30_test_한국',
                                           'over30_test_미국', 'over30_test_전체'])

    row = 0
    for path_name_1, folders_1, files_1 in os.walk(result_root_folder):
        if path_name_1.endswith('_'+str(start_year)):
            if 'spotfire' in path_name_1:
                continue
            for path_name_2, folders_2, files_2 in os.walk(os.path.join(path_name_1, folders_1[0])):

                if 'specifications.txt' in files_2:
                    # path_name_2: D:\Dropbox\DMLab\프로젝트\showbox_2016_09\진원 폴더\result\over_0_미국_관객수_2012\DecisionTreeRegressor\1
                    path = path_name_2.split(os.sep)

                    training_data_name = path[-3]
                    model = path[-2]
                    week = path[-1]
                    mse_score = []

                    with open(os.path.join(path_name_2 ,'specifications.txt'), 'r', encoding='utf-8') as f:
                        specification = f.read()

                        for pattern in patterns:
                            mse = re.search(pattern, specification).group(1)
                            mse_score.append(mse)

                        result_summary.loc[row, 'Training data'] = training_data_name
                        result_summary.loc[row, 'Model'] = model
                        result_summary.loc[row, 'Week'] = week
                        result_summary.loc[row, ['over30_val_한국',
                                                 'over30_val_미국', 'over30_val_전체', 'over30_test_한국',
                                                 'over30_test_미국', 'over30_test_전체']] = mse_score

                        row = row + 1

    result_summary.to_csv(os.path.join(result_root_folder, 'spotfire','{}_result_summary.csv'.format(start_year)),
                          index=False)

if __name__ == '__main__':
    start = '2010-01-01'
    end = '2016-09-04'
    result_fnames = list()
    for nationality in ['전체', '미국', '한국']:
        for cond in [0, 10]:
            result_fnames.append('over_{}_{}_관객수_{}'.format(cond, nationality, start.split('-')[0]))

    spotfire_1(start, end, top_k=100, year_start=int(start.split('-')[0]))
    spotfire_2(start, end, top_k=100)
    spotfire_3(start, end, top_k=100)
    spotfire_4(start, end, top_k=100,
              result_fnames =  result_fnames)
    spotfire_5()

    summarize_results(start.split('-')[0])