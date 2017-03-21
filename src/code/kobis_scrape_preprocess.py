import re

import requests
import os
import csv
import pandas as pd
from bs4 import BeautifulSoup as bs
from datetime import datetime, timedelta
import re
import numpy as np
import multiprocessing
import json

def kobis_scrape_daily(start, end, top_k, seats_or_aud):
    print("start scarping from {} to {}".format(start, end))
    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result', 'daily_seats')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    start = datetime.strptime(start, "%Y-%m-%d").date()
    end = datetime.strptime(end, "%Y-%m-%d").date()

    assert end > start, "end date should be later in time"
    # data_file = open(os.path.join(result_dir, 'daily_box_all_movies_from_{}_to_{}.csv'.format(start, end)), 'w')

    current = start

    header_idx = False
    if seats_or_aud == "aud":
        file_name = 'daily_box_all_movies_from_{}_to_{}_top_{}.csv'.format(start, end, top_k)
    elif seats_or_aud == "seats":
        file_name = 'daily_seats_all_movies_from_{}_to_{}_top_{}.csv'.format(start, end, top_k)

    assert not os.path.exists(os.path.join(result_dir, file_name)), "file exists"

    movie_file = open(os.path.join(result_dir, file_name), "w")
    movie_writer = csv.writer(movie_file)

    while ((current - end).days <= 0):
        print(current)
        if seats_or_aud == "aud":
            url = 'http://www.kobis.or.kr/kobis/business/stat/boxs/findDailyBoxOfficeList.do?loadEnd=0&' \
                  'searchType=excel&sSearchFrom={0}&sSearchTo={0}&' \
                  'sMultiMovieYn=&sRepNationCd=&sWideAreaCd='.format(current)

        elif seats_or_aud == "seats":
            url = "http://www.kobis.or.kr/kobis/business/stat/boxs/findDailySeatTicketList.do?loadEnd=0" \
                             "&totSeatCntRatioOrder=&totSeatCntOrder=&totShowAmtOrder=&addTotShowAmtOrder" \
                             "=&totShowCntOrder=&addTotShowCntOrder=&dmlMode=excel&startDate={0}&endDate={0}" \
                             "&searchType=2&repNationCd=&wideareaCd=".format(current)

        page = requests.get(url)

        soup = bs(page.content, "lxml")

        # print(soup.prettify())

        if header_idx == False:
            rows = soup.find_all('span')

            string = ()
            for row in rows:
                string = string + (row.text.strip(),)
            string = string + ('날짜',)

            movie_writer.writerow(string)

            header_idx = True

        rows = soup.find_all('tr', {'id': True})[:top_k]
        for row in rows:
            elements = row.find_all('td')

            string = ()
            for element in elements:
                string = string + (element.string.strip(),)

            string = string + (str(current),)
            movie_writer.writerow(string)

        movie_file.flush()

        current = current + timedelta(days=1)

    movie_file.close()


def kobis_preprocess(start, end, top_k, weekday, subset = None):
    print("preparing for data")
    print("target weekday", weekday)
    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result', 'daily_seats')
    file_name = 'daily_box_all_movies_from_{}_to_{}_top_{}.csv'.format(start, end, top_k)

    data = pd.read_csv(os.path.join(result_dir, file_name))

    if subset:
        data = data.loc[data["순위"] <= subset, :]
    ## 2010년 이후 데이터만 추출
    after_2010 = [bool(re.match("^20", str(x))) for x in data["개봉일"]]

    for x in data.loc[after_2010, "개봉일"]:
        try:
            datetime.strptime(x, "%Y-%m-%d")
        except ValueError:
            print("doesn't match", x)

    data = data.loc[after_2010]
    data["개봉일"] = [datetime.strptime(x, "%Y-%m-%d") for x in data["개봉일"]]
    data["날짜"] = [datetime.strptime(x, "%Y-%m-%d") for x in data["날짜"]]

    weekday_dic = {"MON" : 0,
                   "TUE" : 1,
                   "WED" : 2,
                   "THU" : 3,
                   "FRI" : 4,
                   "SAT" : 5,
                   "SUN"  : 6}

    weekday_num = weekday_dic[weekday] ##pandas의 weekday는 숫자로
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.weekday.html


    ##개봉 토요일 추출
    data.loc[:, "tf_target"] = [x + timedelta(days=weekday_num - x.weekday()) for x in data.loc[:, "개봉일"]]
    target_data = data.loc[data["tf_target"] == data["날짜"], :]

    #identifier 추가하기
    target_data.loc[:, "Identifier"] = target_data.apply(lambda row : "{}_{}".format(row["영화명"], row["개봉일"].date()),
                                                         axis=1)

    if subset:
        target_data.to_csv(os.path.join(result_dir, "target_data_{}_from_{}_to_{}_top_{}.csv".format(weekday,
                                                                                                        start,
                                                                                                        end,
                                                                                                 subset)), index=False)
    else:
        target_data.to_csv(
            os.path.join(result_dir, "target_data_{}_from_{}_to_{}_top_{}.csv".format(weekday, start,
                                                                                      end,
                                                                                      top_k)), index=False)

    return target_data


def kobis_run(start_date, end_date, top_k, weekday, seats_or_aud):
    print('fetching kobis data')
    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result', "daily_seats")

    processed_file = os.path.join(result_dir,
                                  "target_data_{}_from_{}_to_{}_top_{}.csv".format(weekday, start_date, end_date, top_k))

    if os.path.exists(processed_file):
        return pd.read_csv(processed_file)
    else:
        #없으면 현재 top_k보다 더큰 top_k로 만든게 있다면 거기에서 kobis_preprocess를 돌린다.
        raw_data_file = 'daily_box_all_movies_from_{}_to_{}_top_(\d+).csv'.format(start_date, end_date)
        for file in os.listdir(result_dir):
            match = re.search(raw_data_file, file)
            if match:
                found_top = int(match.group(1))
                if top_k <= found_top:
                    return kobis_preprocess(start_date, end_date, found_top, weekday, subset = top_k)

        # if there is no match
        kobis_scrape_daily(start_date, end_date, top_k, seats_or_aud)
        return kobis_preprocess(start_date, end_date, top_k, weekday)


def kobis_running_time(start, end, top_k, subset):
    print('run time')
    print('subset', subset)
    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result', 'daily_seats')
    aud_file = 'daily_box_all_movies_from_{}_to_{}_top_{}.csv'.format(start, end, top_k)
    aud_data = pd.read_csv(os.path.join(result_dir, aud_file))
    after_2010 = [bool(re.match("^201", str(x))) for x in aud_data["개봉일"]]
    aud_data = aud_data.loc[after_2010]
    aud_data['날짜'] = pd.to_datetime(aud_data['날짜'])
    aud_data['개봉일'] = pd.to_datetime(aud_data['개봉일'])
    aud_data["개봉일차"] = (aud_data["날짜"] - aud_data["개봉일"])
    aud_data["개봉일차"] = [x.days + 1 for x in aud_data["개봉일차"]]

    aud_data["Identifier"] = aud_data.apply(lambda row: "{}_{}".format(row["영화명"], row["개봉일"]),
                                            axis=1)
    # training_data = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'result', 'training_data_관객수.csv'))

    # to_search = [x[:-11] for x in training_data["Identifier"]]
    id = aud_data.drop_duplicates(subset=['영화명', '감독'])

    # for num, i in enumerate(id.index):
    #     try:
    #         movie = id.loc[i, '영화명']
    #     try:
    #         director = id.loc[i, '감독'].split(',')[0]
    #     except AttributeError:
    #         aud_data.loc[i, '상영시간'] = ''
    #         continue
    #     print(movie, director)
    #
    #     r = requests.post('http://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieList.do',
    #                       data={'sMovName': movie, 'sDirector': director})
    #     soup = bs(r.content, 'lxml')
    #     result = soup.find_all('td', {'class': 'ellipsis'})
    #     result2 = result[0].find_all('a')
    #     code = re.search("'movie','(.+)'\)", result2[0]['onclick']).group(1)
    #
    #     r_2 = requests.post('http://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieDtl.do',
    #                         data={'code': code})
    #     soup_2 = bs(r_2.content, 'lxml')
    #     running_time = soup_2.find_all('li')[8].text
    #     try:
    #         aud_data.loc[i, '상영시간'] = int(running_time[:-1])
    #     except ValueError:
    #         aud_data.loc[i, '상영시간'] = ''
    #     if num % 100 == 0:
    #         aud_data.to_csv(os.path.join(result_dir, 'daily_box_all_movies_from_{}_to_{}_top_{}_run.csv'.format(start, end,
    #                                                                                                         top_k)))

    ###### 개봉한 이후 것들만 뽑기 (아니면 개봉전날)
    aud_data = aud_data.loc[aud_data['개봉일차'] >= 0]

    id.sort_index(inplace=True)
    interval = [ range(int(len(id) / 10 * (x-1)), int(len(id) / 10 * x))  for x in [1,2,3,4,5,6,7,8,9,10]]

    for num, i in enumerate(id.index):
        if num in interval[subset]:
            try:
                movie = id.loc[i, '영화명']
                director = id.loc[i, '감독'].split(',')[0]

                # if movie not in to_search:
                #     continue
                r = requests.post('http://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieList.do',
                                  data={'sMovName': movie, 'sDirector': director})
                soup = bs(r.content, 'lxml')
                result = soup.find_all('td', {'class': 'ellipsis'})
                result2 = result[0].find_all('a')
                code = re.search("'movie','(.+)'\)", result2[0]['onclick']).group(1)

                r_2 = requests.post('http://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieDtl.do',
                                    data={'code': code})
                soup_2 = bs(r_2.content, 'lxml')
                running_time = soup_2.find_all('li')[8].text
                aud_data.loc[aud_data['Identifier'] == id.loc[i, 'Identifier'], '상영시간'] = int(running_time[:-1])
                print(movie, director)
            except :
                # if movie in to_search:
                #     print('subset : {},  movie : {}'.format(subset, movie))
                aud_data.loc[aud_data['Identifier'] == id.loc[i, 'Identifier'], '상영시간'] = ''

        # if num % 1 == 0:
        #     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@save@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        #     aud_data.to_csv(
        #         os.path.join(result_dir, 'daily_box_all_movies_from_{}_to_{}_top_{}_run_concise.csv'.format(start, end,
        #                                                                                             top_k)),
        #         index=False)

    condition = aud_data['Identifier'].isin(id.iloc[interval[subset]]['Identifier'])
    aud_data = aud_data.loc[condition]
    aud_data.to_csv(os.path.join(result_dir, 'daily_box_all_movies_from_{}_to_{}_top_{}_run_{}.csv'.format(
        start, end,
                                                                                                     top_k, subset)),
                    index=False)
    return aud_data


def kobis_runtime_merge(start, end, top_k):
    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result', 'daily_seats')

    df_list = [pd.read_csv(os.path.join(result_dir, 'daily_box_all_movies_from_{}_to_{}_top_{}_run_{}.csv'.format(
        start, end, top_k, x))) for x in range(0, 10)]
    df = pd.concat(df_list, ignore_index=True, )

    dup = df.duplicated()
    assert sum(dup) == 0
    df = df.drop_duplicates()
    df.to_csv(os.path.join(result_dir, 'daily_box_all_movies_from_{}_to_{}_top_{}_run_merged.csv'.format(
        start, end, top_k)), index=False)


def _kobis_trend_analysis_data(start, end, top_k):
    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result', 'daily_seats')
    aud_file = 'daily_box_all_movies_from_{}_to_{}_top_{}.csv'.format(start, end, top_k)
    seats_file = 'daily_seats_all_movies_from_{}_to_{}_top_{}.csv'.format(start, end, top_k)

    aud_data = pd.read_csv(os.path.join(result_dir, aud_file))
    seats_data = pd.read_csv(os.path.join(result_dir, seats_file), thousands=',')

    seats_data.drop("순위", axis=1, inplace=True) # 중복됨 그리고 이건 좌점율 순위임

    after_2000 = [bool(re.match("^201", str(x))) for x in aud_data["개봉일"]]
    aud_data = aud_data.loc[after_2000]
    after_2000 = [bool(re.match("^201", str(x))) for x in seats_data["개봉일"]]
    seats_data = seats_data.loc[after_2000]

    aud_data["개봉일"] = pd.to_datetime(aud_data["개봉일"], format = "%Y-%m-%d")
    aud_data["날짜"] = pd.to_datetime(aud_data["날짜"], format = "%Y-%m-%d")
    seats_data["개봉일"] = pd.to_datetime(seats_data["개봉일"], format = "%Y-%m-%d")
    seats_data["날짜"] = pd.to_datetime(seats_data["날짜"], format = "%Y-%m-%d")


    aud_data["Identifier"] = aud_data.apply(lambda row : "{}_{}".format(row["영화명"], row["개봉일"].date()),
                                                         axis=1)

    seats_data["Identifier"] = seats_data.apply(lambda row: "{}_{}".format(row["영화명"], row["개봉일"].date()),
                                            axis=1)

    cols_to_use = list(seats_data.columns.difference(aud_data.columns))
    cols_to_use.extend(["Identifier", "날짜"])

    data = pd.merge(aud_data, seats_data[cols_to_use], on = ["Identifier", "날짜"], how = "left")

    ## derived columns
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

    data.to_csv(os.path.join(result_dir, "daily_aud_seats_merged_from_{}_to_{}.csv".format(start, end)), index=False)


def test_scrape():
    # kobis_run("2012-01-01", "2016-09-04", 100, "SAT")
    # kobis_scrape_daily("2012-01-01", "2016-09-04", 100, seats_or_aud = "seats")
    # kobis_scrape_daily("2010-01-01", "2011-12-31", 100, seats_or_aud="aud")
    kobis_scrape_daily("2010-01-01", "2011-12-31", 100, seats_or_aud="seats")


def test_running_time(start, end, top_k):
    # after kobis_scrape
    # kobis_scrape_daily("2010-01-01", "2011-12-31", 100, seats_or_aud="seats")
    # kobis_scrape_daily("2010-01-01", "2011-12-31", 100, seats_or_aud="aud")

    jobs = []
    for subset in range(10):
        p = multiprocessing.Process(target=kobis_running_time, args=(start, end, top_k, subset))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    #이후에 merge하는거 실행
    kobis_runtime_merge(start, end, top_k)

if __name__ == "__main__":
    # kobis_running_time("2010-01-01", "2016-09-04", 100, 8)
    # kobis_runtime_merge("2010-01-01", "2016-09-04", 100)
    # test_running_time("2012-01-01", "2016-09-04", 100)
    print("2012-01-01", "2016-09-04")
    kobis_scrape_daily("2012-01-01", "2016-09-04", 100, seats_or_aud="seats")