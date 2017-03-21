# -*- coding: utf-8 -*-

'''
Created on 2015. 4. 7.
@author: jinwon
'''
import codecs
import requests
from bs4 import BeautifulSoup as bs
import pdb
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from datetime import timedelta
import re

def scrap_from_web(start, end):
    print('scraping from the web')
    c = requests.Session()
    url = 'http://thelab.kr/login.php'
    id = 'showbox1'
    pw = 'showbox1'

    c.get(url)
    login_data = {'id': id, 'pw': pw}
    c.post(url, data=login_data)

    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result')
    f = open(os.path.join(result_dir, 'funlife_crawling_{0}-{1}.csv'.format(start, end)), 'w')

    column_names = ['영화 제목', '주차',
                    '단순인지도', '단순인지도 %',
                    '유효인지도', '유효인지도 %',
                    '선호도', '선호도 %',
                    '유효선호도', '유효선호도 %',
                    '관람의향', '관람의향 %',
                    '단순인지도증감',
                    '유효인지도증감',
                    '선호도증감',
                    '유효선호도증감',
                    '관람의향증감',
                    '날짜',
                    'Identifier']

    string = u""
    for key in column_names:
        string = string + '"{0}"'.format(key) + ','
    string = string[:-1] + '\n'
    f.write(string)

    data_week_tracker = dict()

    for week_num in np.arange(end-start + 1) + start:
        print(week_num)
        start = time.clock()

        page = c.get('http://thelab.kr/?inc=weekly_coming_movie_all&stage={0}'.format(week_num))
        soup = bs(page.content, "lxml")

        opening_tag = soup.find('option',
                                attrs={'value': '../../?inc=weekly_coming_movie_all&stage={0}'.format(week_num)})
        if not opening_tag:
            print("{} does not exist".format(week_num))
            break
        opening_date = opening_tag.text[-11:-1]

        table_list = soup.find_all('div', attrs={'class': 'table_wrap'})

        for table in table_list:  # 영화마다 하나의 table
            temp_row = dict.fromkeys(column_names)
            temp_row['영화 제목'] = table.p.contents[1].strip()  # movie string 에 trailing whitespace가 있어서

            rows = table.find_all('tr')  # 영화 하나 마다 여러개의 row

            write_list = list()

            week_list = list()
            for row in rows:
                for data in row.find_all('td'):
                    if len(row.attrs) == 0:
                        if data.attrs != {}:
                            week_list.append(int(data.string[:-1]))

            if temp_row['영화 제목'] not in data_week_tracker.keys():
                data_week_tracker[temp_row['영화 제목']] = week_list
                write_list.extend(week_list)
            else:
                for elem in week_list:
                    if elem not in data_week_tracker[temp_row['영화 제목']]:
                        write_list.append(elem)

            min_week = min(week_list)

            add_date = timedelta(weeks=int(min_week) - 1)

            # temp_row['날짜'] = str(datetime(*[int(x) for x in opening_date.split('-')]) + add_date).split(' ')[0]
            temp_row['날짜'] = str((datetime(*[int(x) for x in opening_date.split('-')]) + add_date).date())

            for row in rows:
                if len(row.attrs) == 0:
                    i = 1

                    for data in row.find_all('td'):

                        if data.attrs != {}:  # 주차

                            temp_row[column_names[i]] = '-' + data.string[:-1]  # 숫자만
                            temp_row['Identifier'] = temp_row['영화 제목'] + '_' + temp_row['날짜']
                            i = i + 1
                            continue
                        if data.span.string == None:  # 증감 없을떄

                            temp_row[column_names[i]] = ''
                            i = i + 1
                            continue
                        elif data.string != None:  # 증감 항목일때

                            temp_row[column_names[i]] = data.string
                            i = i + 1
                            continue
                        else:  # 그냥이랑 그냥 %

                            pop = data.contents[0][:-1]
                            percent = data.span.string.strip('()%')
                            temp_row[column_names[i]] = pop
                            temp_row[column_names[i + 1]] = percent

                            i = i + 2
                    string = u""
                    for key in column_names:
                        string = string + '"{0}"'.format(temp_row[key]) + u","
                    string = string[:-1] + u"\n"

                    if int(temp_row['주차'][1:]) in write_list:
                        f.write(string)
        f.flush()
        end = time.clock()
        print('time : {0} seconds'.format(end - start))
    f.close()


def make_horizontal_and_add_relative_vars(start, end):
    print('making horizontal')
    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result')

    column_names = ['영화 제목',
                    'Identifier']

    column_names_idx = ['단순인지도',
                        '단순인지도 %',
                        '유효인지도',
                        '유효인지도 %',
                        '선호도',
                        '선호도 %',
                        '유효선호도',
                        '유효선호도 %',
                        '관람의향',
                        '관람의향 %']

    for col in column_names_idx:
        for week in [-6, -5, -4, -3, -2, -1]:
            column_names.append(col + '_' + str(week))

    df = pd.DataFrame(columns=column_names)

    data = pd.read_csv(os.path.join(result_dir, 'funlife_crawling_{0}-{1}.csv'.format(start, end)))

    data.drop_duplicates(inplace=True)

    pos = 0
    for movie in data['Identifier'].unique():
        df.loc[pos, 'Identifier'] = movie
        df.loc[pos, '영화 제목'] = movie.split('_')[0]
        for col in column_names_idx:
            for week in [-6, -5, -4, -3, -2, -1]:
                a = (data['Identifier'] == movie)
                if data['주차'].dtype == np.int64:
                    b = (data['주차'] == week)
                elif data['주차'].dtype == str:
                    b = (data['주차'] == str(week))
                #             b = (data['주차'] == week)
                val = data.loc[a & b, col]

                if not val.empty:
                    '''
                    for checking : 일일히 print해서 나온 영화들 보정한 the original crawling file
                    '''
                    if len(val) >= 2:
                        print('error : ' + movie)
                        continue

                    df.loc[pos, col + '_' + str(week)] = val.item()

        pos = pos + 1

    # add relative columns
    for week in ["-1", "-2", "-3", "-4", "-5", "-6"]:
        df["유효인지도_상대" + "_" + week] = df["유효인지도" + "_" + week] / df["단순인지도" + "_" + week] * 100
        df["선호도_상대" + "_" + week] = df["선호도" + "_" + week] / df["단순인지도" + "_" + week] * 100
        df["유효선호도_상대" + "_" + week] = df["유효선호도" + "_" + week] / df["단순인지도" + "_" + week] * 100
    # change column name for absolute columns
    new_col = {x: x.replace(" %", "_절대") for x in df.columns if re.search("%", x)}
    df.rename(columns=new_col, inplace=True)

    # add derived columns
    for week in ["-1", "-2", "-3", "-4", "-5", "-6"]:
        df["NET_유효인지도_절대" + "_" + week] = df["유효인지도_절대" + "_" + week] - df["유효선호도_절대" + "_" + week]
        df["NET_선호도_절대" + "_" + week] = df["선호도_절대" + "_" + week] - df["유효선호도_절대" + "_" + week]

        df["NET_유효인지도_상대" + "_" + week] = (df["유효인지도" + "_" + week]  - df["유효선호도" + "_" + week]) \
                                          / df["단순인지도" + "_" + week] * 100

        df["NET_선호도_상대" + "_" + week] = (df["선호도" + "_" + week] - df["유효선호도" + "_" + week]) \
                                          / df["단순인지도" + "_" + week] * 100

        div = [x if x > 0 else 1 for x in df["선호도_절대" + "_" + week]]
        df["NET_선호도_선호도_상대" + "_" + week] = df["NET_선호도_절대" + "_" + week] / div * 100

        div = [x if x > 0 else 1 for x in df["유효인지도" + "_" + week]]
        df["유효선호도_유효인지도_상대" + "_" + week] = df["유효선호도" + "_" + week] / div * 100

        div = [x if x > 0 else 1 for x in df["선호도" + "_" + week]]
        df["유효선호도_선호도_상대" + "_" + week] = df["유효선호도" + "_" + week] / div * 100

    df.to_csv(os.path.join(result_dir, 'funlife_horizontal_{0}-{1}.csv'.format(start, end)), index=False)
    return df


def run_funlife(start, end):
    print('fetching funlife data')
    result_dir = os.path.join(os.path.dirname(os.getcwd()), 'result')
    if os.path.exists(os.path.join(result_dir, 'funlife_horizontal_{0}-{1}.csv'.format(start, end))):
        funlife = pd.read_csv(os.path.join(result_dir, 'funlife_horizontal_{0}-{1}.csv'.format(start, end)))
    elif os.path.exists(os.path.join(result_dir, 'funlife_crawling_{0}-{1}.csv'.format(start, end))):
        funlife = make_horizontal_and_add_relative_vars(start=start, end=end)
    else:
        scrap_from_web(start=start, end=end)
        funlife = make_horizontal_and_add_relative_vars(start=start, end=end)
    return funlife
