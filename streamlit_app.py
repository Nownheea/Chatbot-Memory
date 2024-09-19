import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
import os

# Streamlit 페이지 설정
st.set_page_config(page_title="대화 기억 챗봇", page_icon="🤖")

# OpenAI API 키 설정 (안전한 방법으로 관리해야 합니다)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 제목
st.title("대화 기억 챗봇 🤖")

# 세션 상태 초기화
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
    st.session_state.chain = ConversationChain(
        llm=OpenAI(temperature=0.7),
        memory=st.session_state.memory,
        verbose=True
    )

# 채팅 기록 표시
st.subheader("대화 기록")
for message in st.session_state.memory.chat_memory.messages:
    if message.type == 'human':
        st.text_input("You:", value=message.content, key=f"human_{message.content[:10]}", disabled=True)
    else:
        st.text_area("AI:", value=message.content, key=f"ai_{message.content[:10]}", disabled=True)

# 사용자 입력
user_input = st.text_input("메시지를 입력하세요:", key="user_input")

# 전송 버튼
if st.button("전송"):
    if user_input:
        # AI 응답 생성
        response = st.session_state.chain.predict(input=user_input)
        
        # 화면 갱신
        st.experimental_rerun()

# 대화 초기화 버튼
if st.button("대화 초기화"):
    st.session_state.memory.clear()
    st.success("대화 기록이 초기화되었습니다.")
    st.experimental_rerun()

# 메모리 내용 표시 (디버깅용, 필요시 주석 해제)
# st.write(st.session_state.memory.chat_memory.messages)
