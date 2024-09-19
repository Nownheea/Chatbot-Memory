import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 프롬프트 템플릿 생성
prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     "안녕하세요. 친구처럼 대화하고 이모티콘을 사용해 주세요. 😊"
     "잘 모르겠는 내용에 대해서는 추측하지 말고 '모르겠습니다'라고 대답해 주세요."
     "밝고 긍정적인 태도로 답변해 주세요"),
    ("human", "{input_text}")
])

# 메모리 및 대화 체인 초기화
@st.cache_resource
def initialize_chain():
    memory = ConversationBufferMemory()
    return ConversationChain(
        llm=ChatOpenAI(),
        memory=memory,
        verbose=False
    )

# Streamlit 앱 설정
st.title("AI 채팅 애플리케이션 🤖")

# 대화 기록을 저장할 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 체인 초기화
conversation = initialize_chain()

# 사용자 입력
user_input = st.text_input("메시지를 입력하세요:", key="user_input")

# 메시지 전송 버튼
if st.button("전송"):
    if user_input:
        # 사용자 메시지를 대화 기록에 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # AI 응답 생성
        response = conversation.predict(input=user_input)
        
        # AI 응답을 대화 기록에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # 입력 필드 초기화
        st.session_state.user_input = ""

# 대화 기록 표시
for message in st.session_state.messages:
    if message["role"] == "user":
        st.text_input("You:", value=message["content"], disabled=True)
    else:
        st.text_area("AI:", value=message["content"], disabled=True)

# 대화 기록 초기화 버튼
if st.button("대화 초기화"):
    st.session_state.messages = []
    conversation.memory.clear()
    st.success("대화 기록이 초기화되었습니다.")
