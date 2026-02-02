from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from langgraph.checkpoint.memory import MemorySaver

class ReactAgent:
    def __init__(self):
        #为了支持短期记忆
        self.checkpointer = MemorySaver()

        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            checkpointer=self.checkpointer
        )

    def execute_stream(self, query: str,thread_id = None):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }
        
        # === 修复记忆：配置 thread_id ===
        # 如果传入了 thread_id，LangGraph 就会自动加载历史记忆
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None

        for chunk in self.agent.stream(input_dict, stream_mode="values",config=config):
            latest_message = chunk["messages"][-1]

            if latest_message.type == "human":
                continue

            content = latest_message.content

            #修复，如果最新消息是human发的直接跳过不输出，
            
            # === 兼容 Gemini 修复开始 ===
            # 如果 content 是列表（Gemini 特有），将其转换为字符串
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    # 提取字典中的 'text' 字段，或者是纯字符串元素
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "".join(text_parts)
            # === 兼容 Gemini 修复结束 ===

            if content:
                yield content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
