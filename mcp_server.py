from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import os
from dotenv import load_dotenv

load_dotenv()

async def setup_bright_data_tools():
    """
    Configure Bright Data MCP client and create LangChain-compatible tools
    """
    env_vars = os.environ.copy()
    api_key = os.getenv("BRIGHT_DATA_API_KEY")
    if api_key:
        env_vars["API_TOKEN"] = api_key
    else:
        print("⚠️ Warning: BRIGHT_DATA_API_KEY is missing.")

    try:
        client = MultiServerMCPClient(
            {
                "bright_data": {  
                    "transport": "stdio",
                    "command": "npx",  
                    "args": ["-y", "@brightdata/mcp"], # 参数列表 (-y 自动确认安装，避免阻塞)
                    "env": env_vars
                }
            }
        )
        
        tools = await client.get_tools()
        
        print(f"✅ Connected to Bright Data MCP server")
        print(f"📊 Available tools: {len(tools)}")
        return tools


    except Exception as e:
        print(f"❌ Failed to connect to Bright Data MCP server: {e}")
        return []

if __name__ == "__main__":
    # 需要在 async 环境中运行
    tools = asyncio.run(setup_bright_data_tools())
    for tool in tools:
        print(tool.name)
