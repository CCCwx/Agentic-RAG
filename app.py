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

# Thread ID：同一 thread 内多轮对话；若启用 PostgreSQL 则跨会话持久化
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

# 若配置了 DATABASE_URL，初始化 PostgreSQL 长期记忆表
try:
    from utils.postgres_memory import is_available, create_tables_if_not_exists
    if is_available():
        create_tables_if_not_exists()
except Exception:
    pass

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 对话历史：优先从 PostgreSQL 长期记忆加载，否则用当前 session 消息（不含本条）
    thread_id = st.session_state["thread_id"]
    try:
        from utils.postgres_memory import is_available, get_recent_messages, add_message
        if is_available():
            chat_history = get_recent_messages(thread_id, limit=50)
        else:
            chat_history = st.session_state["message"][:-1]
    except Exception:
        chat_history = st.session_state["message"][:-1]

    response_messages = []
    with st.spinner("RAG 工作流执行中…"):
        res_stream = run_rag_stream(
            prompt,
            thread_id=thread_id,
            chat_history=chat_history,
        )

        def capture(generator, cache_list):
            for char in generator:
                cache_list.append(char)
                time.sleep(0.01)
                yield char

        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        assistant_content = "".join(response_messages)
        st.session_state["message"].append({"role": "assistant", "content": assistant_content})

    # 若启用 PostgreSQL 长期记忆，写入本轮 user 与 assistant 消息
    try:
        from utils.postgres_memory import is_available, add_message
        if is_available():
            add_message(thread_id, "user", prompt)
            add_message(thread_id, "assistant", assistant_content)
    except Exception:
        pass

    st.rerun()
