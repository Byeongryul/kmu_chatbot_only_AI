import pandas as pd
import os

def open_file() :
    path = 'rsc/data_mapping/'
    files = os.listdir(path)
    files_csv = [file for file in files if file.endswith(".csv")]
    datas = {}
    for file_csv in files_csv:
        datas[file_csv.split('.csv')[0]] = pd.read_csv(path + file_csv).drop('Unnamed: 0', axis=1).fillna('')
    return datas

class FindAnswer:
    def __init__(self, db):
        self.db = db
        self.datas = open_file()
        self.result = {}

    def mapping(self):
        for k, v in self.result.items():
            if k in self.datas:
                for row in self.datas[k].T:
                    if self.datas[k]['0'][row] in v:
                        self.result[k] = self.datas[k]['1'][row]
                        break
                    else:
                        self.result[k] = '장학금'

    # 검색 쿼리 생성
    def _make_query(self):
        self.mapping()
        sql = "select url from answer "
        if 'WARD' in self.result:
            sql +=  "where title='" + self.result['WARD']
            if 'Q' in self.result:
                sql += ' ' + self.result['Q']
            sql += "'"
        return sql

    # 답변 검색
    def search(self, ner_predicts):
        if 'WARD' in ner_predicts == False:
            return None
        self.result = ner_predicts
        # 의도명, 개체명으로 답변 검색
        sql = self._make_query()
        print(sql)
        answer = self.db.select_one(sql)
        # 검색되는 답변이 없으면 의도명만 검색
        if answer is None:
            if 'WARD' in self.result:
                ward_name = self.result['WARD']
                self.result = {}
                self.result['WARD'] = ward_name
            sql = self._make_query()
            print(sql)
            answer = self.db.select_one(sql)
        return answer