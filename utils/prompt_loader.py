from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG总结提示词出错，{str(e)}")
        raise e


def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e

#从向量数据库回来后精简知识
def load_knowledge_refinement_prompt():
    path = get_abs_path(prompts_conf["knowledge_refinement_prompt_path"])
    return open(path, "r", encoding="utf-8").read()

#一开始的routing
def load_intent_routing_prompt():
    path = get_abs_path(prompts_conf["intent_routing_prompt_path"])
    return open(path, "r", encoding="utf-8").read()

#重写query
def load_query_expansion_prompt():
    path = get_abs_path(prompts_conf["query_expansion_prompt_path"])
    return open(path, "r", encoding="utf-8").read()

#生成答案时
def load_generation_prompt():
    path = get_abs_path(prompts_conf["generation_prompt_path"])
    return open(path, "r", encoding="utf-8").read()

#utility check
def load_utility_check_prompt():
    path = get_abs_path(prompts_conf["utility_check_prompt_path"])
    return open(path, "r", encoding="utf-8").read()
    
if __name__ == '__main__':
    print(load_report_prompts())



