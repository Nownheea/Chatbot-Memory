import streamlit as st  
from langchain_prompts import ChatPromptTemplate  
from langchain_memory import ConversationBufferMemory  
from langchain_core import ConversationChain, RunnablePassthrough, StrOutputParser  
from langchain_openai import ChatOpenAI  


import os
from getpass import getpass
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# 프롬프트 템플릿 생성  
prompt_template = ChatPromptTemplate.from_messages([  
    ("system",   
     "안녕하세요. 친구처럼 대화하고 이모티콘을 사용해 주세요. 😊"  
     "잘 모르겠는 내용에 대해서는 추측하지 말고 '모르겠습니다' 리고 대답해 주세요."  
     "밝고 긍정적인 태도로 답변해 주세요"),  
    ("human", "{input_text}")  
])  

memory = ConversationBufferMemory()  
conversation = ConversationChain(  
    llm=OpenAI(),  
    memory=memory,  
    verbose=False  
)  

# LangChain 체인 생성  
chain = {"input_text": RunnablePassthrough()} | prompt_template | OpenAI() | StrOutputParser()  

# Streamlit 애플리케이션  
st.title("대화형 AI 챗봇 🗨️")  
st.write("친구처럼 대화해요! 😊")  

# 사용자 입력 받기  
user_input = st.text_input("당신의 질문을 입력하세요:", "")  

# 버튼 클릭 처리  
if st.button("전송"):  
    if user_input:  
        # 예측 호출  
        response = conversation.predict(input=user_input)  
        st.text("AI의 응답: " + response)  
    else:  
        st.warning("질문을 입력해 주세요.")  

st.sidebar.header("사용법")  
st.sidebar.write("질문을 입력하고 '전송' 버튼을 클릭하세요.")  
