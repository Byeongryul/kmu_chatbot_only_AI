import threading
import json

from src.config.DatabaseConfig import *
from src.util.Database import Database
from src.util.BotServer import BotServer
from src.model.intentNerModel import IntentNerModel
from src.util.FindAnswer import FindAnswer

intentNer = IntentNerModel()
def to_client(conn, addr, params):
    db = params['db']
    try:
        db.connect()  # 디비 연결
        # 데이터 수신
        read = conn.recv(2048)  # 수신 데이터가 있을 때 까지 블로킹
        print('===========================')
        print('Connection from: %s' % str(addr))

        if read is None or not read:
            # 클라이언트 연결이 끊어지거나, 오류가 있는 경우
            print('클라이언트 연결 끊어짐')
            exit(0)

        # json 데이터로 변환
        recv_json_data = json.loads(read.decode())
        print("데이터 수신 : ", recv_json_data)
        query = recv_json_data['Query']

        predict = intentNer.input2intentNer(query)
        intent_name = ''
        ner_predicts = {}
        for data in predict:
            key, value = list(data.keys())[0], list(data.values())[0]
            if 'intent' in value: intent_name = key
            else: ner_predicts[key] = ner_predicts[key] + value if key in ner_predicts else value
        #for k, v in temp.items(): ner_predicts.append((v, k)), ner_tags.append(k)
        print(ner_predicts)
        # 답변 검색
        try:
            f = FindAnswer(db)
            answer = f.search(ner_predicts)
        except Exception as ex:
            print(ex)
            answer.url = 'http://' + DB_HOST + '/null/'
        
        send_json_data_str = {
            "Query" : query,
            "Url": url
        }
        message = json.dumps(send_json_data_str)
        conn.send(message.encode())
    except Exception as ex:
        print(ex)
    finally:
        if db is not None: # db 연결 끊기
            db.close()
        conn.close()

if __name__ == '__main__':
    # 질문/답변 학습 디비 연결 객체 생성
    db = Database(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME)
    print("DB 접속")

    port = 7000
    listen = 100

    # 봇 서버 동작
    bot = BotServer(port, listen)
    bot.create_sock()
    print("bot start")

    while True:
        conn, addr = bot.ready_for_client()
        params = {"db": db}
        client = threading.Thread(target=to_client, args=(
            conn, addr, params
        ))
        client.start()