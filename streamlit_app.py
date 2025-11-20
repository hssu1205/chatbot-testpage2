import streamlit as st
from openai import OpenAI

# Page configuration
st.set_page_config(page_title="야식 추천 챗봇", page_icon="🍜")
st.title("야식 추천 챗봇 🍕🥟🍜")
st.caption("OpenAI gpt-4o-mini 기반 · 늦은 밤 무엇을 먹을지 함께 고민해요")

# Load API key from Streamlit secrets
api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.error("OPENAI_API_KEY가 .streamlit/secrets.toml에 설정되지 않았습니다. 파일을 확인해주세요.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# System prompt for the chatbot
SYSTEM_PROMPT = (
    "너는 한국어로 대화하는 야식 추천 전문 챗봇이야. 사용자 기분, 현재 시간, 건강/알레르기/예산/조리 가능 여부를 간단히 파악한 후 "
    "다양한 카테고리(한식, 분식, 중국식, 일본식, 편의점 간편식, 배달 인기 메뉴, 가벼운 건강식)를 균형 있게 2~4개 정도 제안하고 "
    "각 메뉴에 간단한 설명(맛 특징, 칼로리 느낌, 조리/구매 난이도, 대체 옵션)을 1~2문장으로 붙여줘. "
    "사용자가 재료만 있다고 하면 즉석 레시피(최대 5단계)를 제안하고, 너무 기름진 메뉴만 나열하지 않도록 주의해. "
    "대화는 친근하지만 과도한 이모지는 지양하고, 마지막에는 추가로 도와줄 질문을 한 가지 던져 대화를 이어가."
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("야식 관련해서 뭐가 궁금하세요?")

def build_messages():
    return [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages
    ]

def stream_response():
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=build_messages(),
        temperature=0.9,
        top_p=0.95,
        max_tokens=700,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
        if delta:
            yield delta

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_response())
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown(
    """
    <hr style='margin-top:2rem;margin-bottom:0.7rem;'>
    <div style='font-size:0.75rem;color:#999;'>⚠️ 제공되는 정보는 참고용이며, 알레르기나 건강상 특이사항은 반드시 스스로 최종 확인하세요.</div>
    """,
    unsafe_allow_html=True,
)
