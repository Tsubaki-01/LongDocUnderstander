from typing import Optional, List
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from qwen_vl_utils import process_vision_info
from utils.image_encoder import encode_image_base_64
from .base_model import BaseModel


class Qwen_api(BaseModel):
    def __init__(
            self,
            model_name=None,
            api_key=None,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7,
            max_tokens=1024,
            system_prompt=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs
        )

    def create_messages(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> List[dict]:
        """
        创建消息列表

        :param text_input: 文本输入
        :param image_input: 图片输入路径列表
        :return: 消息列表
        """
        pass

    def generate(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> str:
        """
        调用 Qwen模型生成回答

        :param text_input: 文本输入
        :param image_input: 图像输入 （Qwen-2.5 不支持图像，忽略）
        :return: 生成的回答
        """
        self.clean_up()

        messages = self.create_messages(text_input, image_input)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        output_text = response.choices[0].message.content

        self.clean_up()

        return output_text


class Qwen_2_5_api(Qwen_api):
    def __init__(
            self,
            api_key=None,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7,
            max_tokens=2048,
            system_prompt=None,
            **kwargs
    ):
        super().__init__(
            model_name="qwen-plus-1220",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs
        )

    def create_messages(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> List[dict]:
        """
        创建消息列表

        :param text_input: 文本输入
        :param image_input: 图像输入 （Qwen-2.5 不支持图像，忽略）
        :return: 消息列表
        """
        messages = [
            {
                "role": "system",
                "content":
                    [
                        {
                            "type": "text",
                            "text": self.system_prompt
                        }
                    ]
            },
            {
                "role": "user",
                "content":
                    [
                        {
                            "type": "text",
                            "text": "\n\n" + "**Input:** \n"
                                    + text_input
                        }
                    ]
            }
        ]

        return messages



class Qwen_VL_2_5_api(Qwen_api):
    def __init__(
            self,
            api_key=None,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7,
            max_tokens=4096,
            system_prompt=None,
            **kwargs
    ):
        super().__init__(
            model_name="qwen2.5-vl-32b-instruct",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs
        )

    def create_messages(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> List[dict]:
        """
        创建消息列表

        :param text_input: 文本输入
        :param image_input: 图片输入路径列表
        :return: 消息列表
        """
        messages = [
            {
                "role": "system",
                "content":
                    [
                        {
                            "type": "text",
                            "text": self.system_prompt
                        }
                    ]
            },
            {
                "role": "user",
                "content":
                    [
                        {
                            "type": "text",
                            "text": self.system_prompt + "\n\n" + "**Input:** \n"
                                    + text_input
                        }
                    ]
            }
        ]

        for image_path in image_input:
            image_base64 = encode_image_base_64(image_path)
            image_format = image_path[image_path.rfind('.') + 1:]
            messages[1]["content"].append(
                {
                    "type": "image_url",
                    "image_url":
                        {
                            "url": f"data:image/{image_format};base64,{image_base64}"
                        }
                }
            )

        return messages


class Qwen(BaseModel):
    def __init__(
            self,
            model_name=None,
            api_key=None,
            base_url=None,
            temperature=0.7,
            max_tokens=2048,
            system_prompt=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs
        )
        
        
    def create_messages(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> List[dict]:
        """
        创建消息列表

        :param text_input: 文本输入
        :param image_input: 图片输入路径列表
        :return: 消息列表
        """
        pass
    
    def generate(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> str:
        """
        调用 Qwen 模型生成回答

        :param text_input: 文本输入
        :param image_input: 图像输入 （Qwen-2.5 不支持图像，忽略）
        :return: 生成的回答
        """
        self.clean_up()

        messages = self.create_messages(text_input, image_input)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image, video = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image,
            videos=video,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        self.clean_up()

        return output_text
    
    
class Qwen_VL_2_5(Qwen):
    def __init__(
            self,
            api_key=None,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7,
            max_tokens=2048,
            system_prompt=None,
            **kwargs
    ):
        super().__init__(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs
        )
        max_pixels = 2048*28*28 // 2 // 2 
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/data/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/5b5eecc7efc2c3e86839993f2689bbbdf06bd8d4", torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            "/data/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/5b5eecc7efc2c3e86839993f2689bbbdf06bd8d4",
            max_pixels=max_pixels)

    def create_messages(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> List[dict]:
        """
        创建消息列表

        :param text_input: 文本输入
        :param image_input: 图片输入路径列表
        :return: 消息列表
        """
        messages = [
            {
                "role": "system",
                "content":
                    [
                        {
                            "type": "text",
                            "text": self.system_prompt
                        }
                    ]
            },
            {
                "role": "user",
                "content":
                    [
                        {
                            "type": "text",
                            "text": self.system_prompt + "\n\n" + "**Input:** \n"
                                    + text_input
                        }
                    ]
            }
        ]

        for image_path in image_input:
            image_base64 = encode_image_base_64(image_path)
            image_format = image_path[image_path.rfind('.') + 1:]
            messages[1]["content"].append(
                {
                    "type": "image",
                    "image": image_path
                }
            )

        return messages