import time
import streamlit as st
from agent.react_agent import ReactAgent
import uuid # <--这个是为了config number
from dotenv import load_dotenv
load_dotenv()
# 标题
st.title("Your Personal Agentic Assistant")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

# === 修改点 A：初始化 Thread ID (短期记忆的关键) ===
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("Thinking"):
        res_stream = st.session_state["agent"].execute_stream(
            prompt, 
            thread_id=st.session_state["thread_id"]
        )

        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()
