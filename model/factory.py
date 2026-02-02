from abc import ABC, abstractmethod
from typing import Optional, Union
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# 引入 Google GenAI 的相关库
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from utils.config_handler import rag_conf

import os
from dotenv import load_dotenv
load_dotenv()


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        # 使用 ChatGoogleGenerativeAI
        # 注意：temperature 等参数可以在这里添加，或者让其使用默认值
        return ChatGoogleGenerativeAI(
            model=rag_conf["chat_model_name"],
            # 如果需要调整生成时的随机性，可以取消下面注释
            # temperature=0,
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings]:
        # 使用 GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=rag_conf["embedding_model_name"]
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()