import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 채팅봇", page_icon="🤖")

# OpenAI API 키 설정 (보안을 위해 환경 변수나 Streamlit의 secrets 관리를 사용하는 것이 좋습니다)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# openai.api_key = st.secrets["openai"]["api_key"]

# 프롬프트 템플릿 생성
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "안녕하세요. 친구처럼 대화하고 이모티콘을 사용해 주세요. 😊"
     "잘 모르겠는 내용에 대해서는 추측하지 말고 '모르겠습니다'라고 대답해 주세요."
     "밝고 긍정적인 태도로 답변해 주세요"),
    ("human", "{input}")
])

# LangChain 체인 생성
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=ChatOpenAI(),
    memory=memory,
    prompt=prompt_template,
    verbose=False
)

# Streamlit UI
st.title("AI 채팅봇과 대화하기 🤖")

# 세션 상태로 대화 기록 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

# 사용자 입력
user_input = st.text_input("메시지를 입력하세요:")

if user_input:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # AI 응답 생성
    response = conversation.predict(input=user_input)
    
    # AI 응답 추가
    st.session_state.messages.append({"role": "assistant", "content": response})

# 대화 기록 표시
for message in st.session_state.messages:
    if message["role"] == "user":
        st.text_area("You:", value=message["content"], height=50, disabled=True)
    else:
        st.text_area("AI:", value=message["content"], height=100, disabled=True)

# 대화 초기화 버튼
if st.button("대화 초기화"):
    st.session_state.messages = []
    memory.clear()
    st.success("대화가 초기화되었습니다.")
if st.button("대화 초기화"):
    st.session_state.memory.clear()
    st.success("대화 기록이 초기화되었습니다.")
    st.experimental_rerun()

# 메모리 내용 표시 (디버깅용, 필요시 주석 해제)
# st.write(st.session_state.memory.chat_memory.messages)
