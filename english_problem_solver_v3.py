# English Problem Solver v3
# 기능1. 텍스트 입력하면 gpt-4-turbo가 답변

# 기능2. 이미지 업로드하면 gpt-4-vision이 답변
# - 그래프 이미지 업로드
# - 문제 전체 이미지 업로드

# share streamlit 에서 배포 성공

import os
import base64
import requests
from IPython.display import Image
from PIL import Image as Img
import pandas as pd
import numpy as np
import time
from datetime import datetime

import streamlit as st
import openai
from openai import OpenAI
# from dotenv import load_dotenv
# load_dotenv()
# openai.api_key = os.environ.get('OPENAI_API_KEY')

openai.api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI()

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

st.header("📄English Problem Solver📊")

# 사이드에 list의 선택박스를 생성한다.
select = st.sidebar.selectbox('메뉴', ['텍스트 분석', '그래프 이미지 분석'])

if select == '텍스트 분석':
    # 사용자 질문 입력
    question = st.text_area("문제를 입력해주세요", height=10)

    # 메시지 히스토리 생성
    messages=[{"role": "system", "content": """
               You are a helpful assistant. 
               Answer the questions given and explain your reasoning.
               You must answer in Korean."""}]

    question_dict = {
                "role": "user",
                "content": question ,  # 첫 번째 질문
            }

    # 메시지 히스토리에 질문 추가
    messages.append(question_dict)

    # 버튼 누르면 답변 받기 시작
    if st.button('풀어줘!'):
    # st.write('Why hello there')

        # 답변 받기
        completion = client.chat.completions.create(
            model="gpt-4-0125-preview",          # "gpt-3.5-turbo-0125", "gpt-4-0125-preview"
            messages = messages,
        )

        response = completion.choices[0].message.content

        st.info(response)

if select == '그래프 이미지 분석':

    # 그래프 이미지 업로드
    st.text("")
    st.text("")
    st.write("그래프 이미지를 입력해주세요.")
    graph_image_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], key="graph_image")

    # 문제 전체 이미지 입력
    st.text("")
    st.text("")
    st.write("전체 문제 이미지를 입력해주세요.")
    full_image_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], key="full_image")

    if full_image_file is not None:    # 만약 그래프 이미지 파일과 지문 텍스트가 업로드되면,
        st.image(full_image_file, width=500)   # 전체 문제 이미지 출력

        # 버튼 누르면 gpt에게 답변 받기 시작
        if st.button('풀어줘!'):

            # --------------------------deplot으로 수치 추출하기-------------------------

            processor = Pix2StructProcessor.from_pretrained('nuua/ko-deplot')
            model = Pix2StructForConditionalGeneration.from_pretrained('nuua/ko-deplot')
            
            image = Img.open(graph_image_file)
            inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
            predictions = model.generate(**inputs, max_new_tokens=512)
            result = processor.decode(predictions[0], skip_special_tokens=True)
            # result = ""

            st.text(result)

            # --------------------------gpt-4-vision으로 문제 풀기--------------------------

            # 메시지 히스토리 생성
            messages=[{"role": "system", "content": """
                       you're a mathematician who calculates numbers correctly and makes comparisons with rigor and precision.
                       choose the one you think is the most incorrect sentence. you must answer in Korean.
                       """}]

            # 첫번째 질문 생성
            question = f""" 
            I give you the results of the graph analysis.
            result : {result}.
            Q. Read the following passage and find the sentence that does not match the figure in the chart. 
            """ 

            full_image_path = full_image_file.name

            # Function to encode the image
            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
                    
            # Getting the base64 string
            base64_image = encode_image(full_image_path)

            question_dict = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                        },
                    ],
                    }

            # 메시지 히스토리에 질문 추가
            messages.append(question_dict)

            # 답변 받기
            completion = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages = messages,
            )

            response = completion.choices[0].message.content

            st.info(response)

    else:
        st.info('☝️ 그래프 문제를 업로드하면 문제를 풀어드립니다.')


