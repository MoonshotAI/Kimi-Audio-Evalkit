import torch
import librosa
import json
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

from .base import BaseModel
from .fun_audio_chat.utils.constant import (
    DEFAULT_S2T_PROMPT,
    AUDIO_TEMPLATE,
    FUNCTION_CALLING_PROMPT
)
from .fun_audio_chat.funaudiochat.register import register_funaudiochat
register_funaudiochat()

class FunAudioChat(BaseModel):
    NAME = 'Fun-Audio-Chat'

    def __init__(self, model_path="pretrains/Fun-Audio-Chat-8B", **kwargs):
        super().__init__()

        config = AutoConfig.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, device_map="auto")

        # 生成参数
        self.model.sp_gen_kwargs.update({
            'text_greedy': True, 
            'disable_speech': True,
        })

    @staticmethod
    def check_audio_legal(audio_path: str | list[str], max_duration: float = 60) -> bool:
        """ Override this method to support audio longer than sixty seconds. """
        if isinstance(audio_path, str):
            duration = librosa.get_duration(path=audio_path)
            if duration < 0.1:
                return False
        else:
            for path in audio_path:
                duration = librosa.get_duration(path=path)
                if duration < 0.1:
                    return False
        return True

    def get_prompt(self, msg: dict):
        meta = msg['meta']
        if meta['task'] == 'ASR':
            # from https://github.com/FunAudioLLM/Fun-Audio-Chat#ASR
            prompt = "Please help me transcribe the audio."
        else:
            prompt = msg['text']
        return prompt

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        if len(audio) == 1:
            audio = [librosa.load(audio[0], sr=16000)[0]]
        else:
            raise NotImplementedError(
                f'Audio length {len(audio)} not supported')
        prompt = self.get_prompt(msg)

        if "function" in msg and msg["function"]:
            tools = msg["function"]
            tools_definition = "\n".join([json.dumps(tool_item, ensure_ascii=False) for tool_item in tools])
            system_prompt = FUNCTION_CALLING_PROMPT.replace("{tools_definition}", tools_definition)
            if "time" in msg and msg["time"]:
                system_prompt += f"\nNow Time: {msg['time']}"
        else:
            system_prompt = DEFAULT_S2T_PROMPT
        if prompt is None or prompt.strip() == '':
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": AUDIO_TEMPLATE}
            ]
        else:
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": AUDIO_TEMPLATE + "\n" + prompt}
            ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(self.model.device)
        generate_ids, _ = self.model.generate(**inputs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        generate_text = self.processor.decode(generate_ids[0], skip_special_tokens=True)
        return prompt, generate_text
