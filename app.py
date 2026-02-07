import time
import streamlit as st
from agent.corrective_graph import run_rag_stream
import uuid
from dotenv import load_dotenv
load_dotenv()

# 标题：Robust Agentic RAG v2.0
st.title("Robust Agentic RAG Assistant")
st.divider()

if "use_rag_graph" not in st.session_state:
    st.session_state["use_rag_graph"] = True

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
    with st.spinner("RAG 工作流执行中…"):
        res_stream = run_rag_stream(prompt, thread_id=st.session_state["thread_id"])

        def capture(generator, cache_list):
            for char in generator:
                cache_list.append(char)
                time.sleep(0.01)
                yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        st.session_state["message"].append({"role": "assistant", "content": "".join(response_messages)})
        st.rerun()
