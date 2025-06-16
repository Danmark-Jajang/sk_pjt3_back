from flask import Flask, jsonify, request, render_template
import openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, GPTVectorStoreIndex
import faiss
from llama_index.core import StorageContext, load_index_from_storage
from groq import Groq
from openai import OpenAI
import requests
from datetime import datetime, timedelta
import re
import json
import os
import jinja2
from flask_cors import CORS
from dotenv import load_dotenv
from kakao import kakao_bp

# .env 파일 로드
load_dotenv()

# API Key 설정

# 환경 변수에서 API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # 모든 도메인에서 접근 허용

app.config['JSON_AS_ASCII'] = False
app.register_blueprint(kakao_bp)

# index dump 없을 때 실행시키는 코드
# document = SimpleDirectoryReader('./data').load_data()
# index = GPTVectorStoreIndex.from_documents(document)
# index.storage_context.persist('index_db_backup')

# 기상청 API Base URL
SHORT_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
MID_URL = "http://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst"

# 서울 지역 기본 좌표 -> 고정값 변경X (지역을 전국으로 확대한다면 동적으로 변경 필요)
SEOUL_NX, SEOUL_NY = 60, 127

def get_latest_valid_base_time():
    """ 사용 가능한 최근 `base_time` 반환 (3시간 단위) """
    now = datetime.now()
    base_date = now.strftime("%Y%m%d")

    valid_hours = [2, 5, 8, 11, 14, 17, 20, 23]
    latest_hour = max([h for h in valid_hours if h <= now.hour])
    base_time = f"{latest_hour:02d}00"

    return base_date, base_time

def fetch_short_weather(region, user_date):
    """ 단기예보 (3일 이내 예보) 데이터 조회 """
    nx, ny = SEOUL_NX, SEOUL_NY
    base_date, base_time = get_latest_valid_base_time()

    params = {
        "serviceKey": WEATHER_API_KEY,
        "numOfRows": 1000,
        "pageNo": 1,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny
    }

    response = requests.get(SHORT_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if "response" in data and "body" in data["response"] and "items" in data["response"]["body"]:
            items = data["response"]["body"]["items"].get("item", [])
            filtered_data = [d for d in items if d["fcstDate"] == user_date]

            if filtered_data:
                weather_summary = {d["category"]: d["fcstValue"] for d in filtered_data}

                return {
                    "temperature": f"{weather_summary.get('TMP', 'N/A')}°C",
                    "precipitation": {
                        "0": "없음", "1": "비", "2": "비/눈", "3": "눈"
                    }.get(weather_summary.get('PTY', "0"), "N/A"),
                    "condition": {
                        "1": "맑음", "3": "구름 많음", "4": "흐림"
                    }.get(weather_summary.get('SKY', "1"), "N/A")
                }

    return {"temperature": "데이터 없음", "precipitation": "데이터 없음", "condition": "데이터 없음"}

def fetch_mid_weather(region, user_date):
    """ 중기예보 (4~10일 후 예보) 데이터 조회 """
    now = datetime.now()
    today = now.strftime("%Y%m%d")

    date_obj = datetime.strptime(user_date, "%Y%m%d")
    days_ahead = (date_obj - now).days

    if days_ahead < 4:
        return None

    params = {
        "serviceKey": WEATHER_API_KEY,
        "numOfRows": 10,
        "pageNo": 1,
        "dataType": "JSON",
        "regId": "11B00000",
        "tmFc": today + "0600"
    }

    response = requests.get(MID_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if "response" in data and "body" in data["response"] and "items" in data["response"]["body"]:
            items = data["response"]["body"]["items"].get("item", [])
            if items:
                forecast = items[0]
                forecast_key = f"rnSt{days_ahead}Am"
                rain_prob = forecast.get(forecast_key, "N/A")

                return {
                    "temperature": "N/A (중기 예보는 기온 정보 없음)",
                    "precipitation": f"{rain_prob}%",
                    "condition": "비 예보 있음" if rain_prob and int(rain_prob) > 50 else "맑음"
                }

    return None

def fetch_weather(region, user_date):
    """ 사용자가 요청한 날짜에 맞춰 단기예보 또는 중기예보를 조회 """
    now = datetime.now()
    days_ahead = (datetime.strptime(user_date, "%Y%m%d") - now).days

    if days_ahead >= 4:
        mid_weather = fetch_mid_weather(region, user_date)
        if mid_weather:
            return mid_weather

    return fetch_short_weather(region, user_date)


# Faiss Vector DB로 실행시키는 코드
# 왜인지는 모르겠지만 계속 에러뜸, 사용금지;;;;
# faiss_index = faiss.IndexFlatL2(1536)
# document = SimpleDirectoryReader('./data').load_data()
# vectorstore = FaissVectorStore(faiss_index=faiss_index)
# storage_context = StorageContext.from_defaults(vector_store=vectorstore)
# index = GPTVectorStoreIndex.from_documents(
#     document,
#     storage_context = storage_context
# )
# index.storage_context.persist('index_db_backup')


# 이미 생성된 VectorDB 내용 가져오기
storage_context = StorageContext.from_defaults(persist_dir='./index_db_backup')
index    = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()



client_groq = Groq(
    api_key=os.environ.get(GROQ_API_KEY)
)

# 키워드를 추출하는 함수
def extract_region_and_keywords(text):
    # 예시: 한국 주요 도시 및 키워드 패턴 (프로젝트에 맞게 확장 가능)
    regions = ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랴구", "성북구", "강북구",
                "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구",
                  "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"]
    keywords = ["문화시설", "축제", "공연", "행사", "관광지", "여행코스", "레포츠", "숙박", "쇼핑", "음식점"]
    types = ["근처", "인근", "주변", "사람 많은", "사람 적은", "유명한", "조용한", "인기 많은"]

    region = next((r for r in regions if r in text), None)
    keyword = next((k for k in keywords if k in text), None)
    types = next((t for t in types if t in text), None)

    return region, keyword, types

# Groq 답변
def tour_query_korean(query):
  res = query_engine.query(query)
  sys_prompt = f'''
지침:
- 도움이 되고 간결하게 답할 것. 답을 모르면 '잘 모르겠어요'라고 말할 것
- 정확하고 구체적인 정보를 얻기 위해 제공된 맥락을 활용할 것
- 기존 지식을 통합하여 답변의 깊이와 관련성을 높일 것
- 출처를 밝힐 것
- 답변은 반드시 한국어로 할 것
- 사용자가 질문한 지역과 일치하는 지역만 검색할 것
- 아래 적힌 내용만 사용해서 검색할 것

내용: {res}
'''

  completion = client_groq.chat.completions.create(
      model = 'llama3-8b-8192',
      messages = [
          {
              'role': 'system',
              'content': sys_prompt
          },
          {
              'role':'user',
              'content': query
          }
      ]
  )
  print('index query: ',res)
  return completion.choices[0].message.content



client_openai = OpenAI(
    api_key=os.environ.get(OPENAI_API_KEY)
)

# OpenAI 답변
def tour_query_openai_korean(query):
  res = query_engine.query(query)
  print('index query: ',res)
  sys_prompt = f'''
지침:
- 너는 도움되는 여행 플래너이다.
- 도움이 되고 자세하게 답할 것. 답을 모르면 '잘 모르겠어요'라고 말할 것
- 정확하고 구체적인 정보를 얻기 위해 제공된 맥락을 활용할 것
- 기존 지식을 통합하여 답변의 깊이와 관련성을 높일 것
- 출처를 밝힐 것
- 답변은 반드시 한국어로 할 것
- 사용자가 질문한 지역과 일치하는 지역만 검색할 것
- 아래 적힌 내용만 사용해서 검색할 것
내용: {res}
'''
  answer = client_openai.chat.completions.create(
      model = 'gpt-4o-mini',
      messages = [
          {
              'role': 'system',
              'content': sys_prompt
          },
          {
              'role':'user',
              'content': query
          }
      ],
      temperature=0.6,
      max_tokens=500
  )
  return answer.choices[0].message.content

import re
from datetime import datetime, timedelta

def extract_date_from_query(query):
    """
    사용자의 입력에서 날짜(YYYYMMDD)를 추출하는 함수
    - '22일'처럼 날짜만 입력하면 현재 연도/월을 자동 적용
    - 'YYYY년 MM월 DD일' 형식도 처리 가능
    - '내일', '모레', '이번 주말' 같은 표현도 처리
    """
    now = datetime.now()
    current_year, current_month, current_day = now.year, now.month, now.day

    # 정규식으로 'YYYY년 MM월 DD일' 패턴 찾기
    full_date_match = re.search(r'(\d{4})년 (\d{1,2})월 (\d{1,2})일', query)

    # '22일' 같은 단순 날짜 패턴 찾기
    day_match = re.search(r'(\d{1,2})일', query)

    if full_date_match:
        year, month, day = map(int, full_date_match.groups())
        return f"{year}{month:02d}{day:02d}"

    if day_match:
        day = int(day_match.group(1))
        # 이미 지난 날짜면 다음 달로 이동
        if day < current_day:
            if current_month == 12:
                target_date = datetime(current_year + 1, 1, day)
            else:
                target_date = datetime(current_year, current_month + 1, day)
        else:
            target_date = datetime(current_year, current_month, day)
        return target_date.strftime("%Y%m%d")

    # 자연어 날짜 표현 처리
    natural_dates = {
        "내일": timedelta(days=1),
        "모레": timedelta(days=2),
        "이번 주말": timedelta(days=(5 - now.weekday()) if now.weekday() < 5 else 1),
        "다음 주말": timedelta(days=(12 - now.weekday()) if now.weekday() < 5 else 8),
        "다음주": timedelta(days=7),
        "다음 주": timedelta(days=7),
    }

    for key, delta in natural_dates.items():
        if key in query:
            return (now + delta).strftime("%Y%m%d")

    return None  # 날짜가 없는 경우


def tour_query_openai_korean_jinja2(query):
    """기상 정보 반영하여 여행 일정 추천 """

    # 사용자 요청 날짜 추출
    user_date = extract_date_from_query(query) or datetime.now().strftime("%Y%m%d")
    today_date = datetime.now().strftime("%Y-%m-%d")
    print(f"[LOG] 사용자가 요청한 날짜: {user_date}")
    print(f"[LOG] 오늘 날짜: {today_date}")

    # 사용자 입력에서 지역, 키워드, 유형 추출
    region, keyword, types = extract_region_and_keywords(query)
    print(f"🔍 추출된 키워드: 지역={region}, 키워드={keyword}, 유형={types}")

    # 벡터 DB에서 여행 관련 데이터 검색 -> 제일 중요함
    res = query_engine.query(query)
    print(f" [벡터 DB 검색 결과]: {res}")

    # 해당 날짜의 기상 정보 가져오기
    weather_data = fetch_weather(region if region else "서울", user_date)
    print(f" [날씨 데이터]: {weather_data}")

    # 날씨 상태 기반 추천 로직
    weather_condition = weather_data["condition"]
    precipitation = weather_data["precipitation"]

    if precipitation and precipitation.endswith("%"):
        weather_impact = f"{user_date}에는 강수 확률이 {precipitation}입니다."
    elif precipitation in ["비", "비/눈", "눈"]:
        weather_impact = "해당 날짜에는 비 또는 눈이 예상됩니다. 실내 관광지를 추천하겠습니다."
    elif weather_condition == "구름 많음" or weather_condition == "흐림":
        weather_impact = "흐린 날씨입니다. 실내외 관광지를 적절히 섞어 추천하겠습니다."
    else:
        weather_impact = "맑은 날씨입니다! 야외 활동하기 좋은 날이네요."

    # OpenAI 프롬프트 구성 (기존 검색 데이터 + 날씨 데이터 반영)
    sys_prompt = f"""
    지침:
    - 오늘 날짜는 {today_date}입니다.
    - 너는 도움되는 여행 플래너이다.
    - 도움이 되고 자세하게 답할 것. 답을 모르면 '잘 모르겠어요'라고 말할 것
    - 정확하고 구체적인 정보를 얻기 위해 제공된 맥락을 활용할 것
    - 기존 지식을 통합하여 답변의 깊이와 관련성을 높일 것
    - 출처를 밝힐 것
    - 답변은 반드시 한국어로 할 것
    - 사용자가 요청한 날짜({user_date})의 날씨 정보를 활용하여 일정 추천할 것.
    - 현재 날씨 정보: {weather_data}
    - {weather_impact}
    - 검색된 여행 정보:
        {res}
    - 장소 이름은 반드시 **굵게(`**`)** 표기할 것. 예: **경복궁**, **남산타워**
    - 장소 이름 외에는 절대 **기호(`**`)**를 사용하지 말 것.
    - 강조나 다른 표현에서 `**`는 절대 사용하지 말 것.
    - 추천한 장소에서 갈 수 있는 여행 경로도 추천할 것.
    - 추천 시 사용자의 취향을 반영할 것.
    - 장소에 기반하여 근처에 갈 수 있는 취향에 맞는 관광지들을 묶어 여행 계획을 추천할 것.
    - 사용자의 취향을 반영하지 않을 때에는 유명한 장소 위주로 추천할 것.
    - 집중률이 높은 지역은 사람이 많고 유명한 관광지로 판별할 것
    - 집중률이 낮은 지역은 사람이 적고 덜 알려진 관광지로 판별할 것
    - 사용자의 취향을 반영할 때 사람이 많거나 적은 관광지에서 따라올 수 있는 상황을 고려할 것
    - 집중률이 높은 지역이라도 관광지의 특성을 반영하여 상황을 판단할 것
    - 아래 적힌 내용만 사용해서 검색할 것
    주요내용: {region if region is not None else ""}, {keyword if keyword is not None else ""}, {types if types is not None else ""}
    내용: {res}
    """

    # OpenAI GPT-4o-mini 호출
    answer = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.6,
        max_tokens=1000
    )

    return answer.choices[0].message.content



# 이미지 생성
def create_image(prompt):
  summary = summarize_query_openai(prompt)
  img =  client_openai.images.generate(
    model = 'dall-e-3',
    prompt = summary,
    size = '1024x1024',
    quality = 'standard',
    n=1
  )
  return img.data[0].url

# 내용요약
def summarize_query_openai(query):
  sys_prompt = f"""
지침
- 너는 주어진 프롬프트에서 중요한 키워드만 뽑아서 요약한 프롬프트를 만들기 위한 agent이다
- 요약된 결과는 반드시 100자 이내여야 한다
- 여행 관련된 키워드로 뽑아야 한다
- 관광지와 관련된 키워드로 뽑아야 한다
"""
  summary = client_openai.chat.completions.create(
    model = 'gpt-4o-mini',
      messages = [
          {
              'role': 'system',
              'content': sys_prompt
          },
          {
              'role':'user',
              'content': query
          }
      ],
      temperature=0.8,
      max_tokens=100
  )
  return summary.choices[0].message.content

# Groq 챗봇 query
# 쓰지마세요@@@@@
# @app.route('/api/post/groq', methods=['POST'])
# def query_groq_post():
#   request_data = request.get_json()
#   query = request_data["query"]
#   return json.dumps({'query':tour_query_korean(query)}, ensure_ascii=False)

# OpenAI 챗봇 query
# 쓰지마세요@@@@@
@app.route('/api/post/openai', methods=['POST'])
def query_openai_post():
  request_data = request.get_json()
  query = request_data["query"]
  return json.dumps({'query':tour_query_openai_korean(query)}, ensure_ascii=False)

# OpenAI 챗봇 query V2
@app.route('/api/post/openai/v2', methods=['POST'])
def query_openai_post_v2():
  request_data = request.get_json()
  query = request_data["query"]
  return json.dumps({'query':tour_query_openai_korean_jinja2(query)}, ensure_ascii=False)

# Dall-e-3 포스터 생성
@app.route('/api/post/openai/poster', methods=['POST'])
def query_openai_poster():
  request_data = request.get_json()
  query = request_data['query']
  return json.dumps({'images':create_image(query)})

# Flask 실행
if __name__ == '__main__':
  app.run(debug=True)