import yaml
from pathlib import Path


def prompt_loader(agent_or_model_name: str) -> str:
    """
    从 config/prompts.yaml 中加载指定 agent 或 model 的 system_prompt

    :param agent_or_model_name: 如 "decompose_agent"
    :return: system_prompt 字符串
    """
    try:
        prompt_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data[agent_or_model_name]["system_prompt"]
    except KeyError:
        raise ValueError(f"找不到 {agent_or_model_name} 的 system_prompt 配置")
    except Exception as e:
        raise RuntimeError(f"加载 system_prompt 失败: {e}")


def api_key_loader(model_name: str) -> str:
    """
    从 config/api-keys.yaml 中加载指定 agent 或 model 的 system_prompt

    :param model_name: 如 "qwen"
    :return: system_prompt 字符串
    """
    try:
        prompt_path = Path(__file__).parent.parent / "config" / "api-keys.yaml"
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data[model_name]["api_key"]
    except KeyError:
        raise ValueError(f"找不到 {model_name} 的 api_key 配置")
    except Exception as e:
        raise RuntimeError(f"获取 api_key 失败: {e}")
