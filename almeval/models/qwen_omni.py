import random

import torch
from qwen_omni_utils import process_mm_info
from transformers import (Qwen2_5OmniForConditionalGeneration,
                          Qwen2_5OmniProcessor)

from ..utils.misc import print_once
from .base import BaseModel


class Qwen2_5Omni(BaseModel):
    NAME = 'Qwen2.5-Omni-7B'

    def __init__(self, model_path='Qwen/Qwen2.5-Omni-7B', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path
        )

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path, device_map='cuda').eval()
        random.seed(0)
        torch.cuda.empty_cache()

    def get_prompt(self, msg: dict):
        # according to https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb
        meta = msg['meta']
        if meta['task'] == 'ASR':
            assert 'lang' in meta
            lang = meta['lang']
            if lang == 'zh':
                prompt = '请将这段中文语音转换为纯文本，去掉标点符号。'
            elif lang == 'en':
                prompt = 'Transcribe the English audio into text without any punctuation marks.'
            else:
                raise NotImplementedError
        elif meta['dataset_name'] == 'vocalsound':
            # from https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb
            prompt = 'Classify the given human vocal sound in English.'
        elif meta['dataset_name'] == 'meld':
            # from: https://github.com/QwenLM/Qwen2-Audio/blob/main/eval_audio/EVALUATION.md
            prompt = 'Recognize the emotion with keywords in English.'
        elif meta['audio_type'] == 'AudioEvent':
            prompt = f'Listen to the given audio carefully and answer this question: {msg["text"]}.'
        else:
            prompt = msg['text']
        return prompt

    def get_system_prompt(self, msg: dict):
        meta = msg['meta']
        if meta is None:
            return ''
        # from: https://github.com/QwenLM/Qwen2.5-Omni/blob/6c1784249f8aa498a0893ec442e20557c2fa5773/web_demo.py#L41C29-L41C192
        system_prompt = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
        if meta['task'] == 'ASR':
            # from: https://github.com/QwenLM/Qwen2.5-Omni/blob/main/cookbooks/universal_audio_understanding.ipynb
            system_prompt = 'You are a speech recognition model.'
        elif meta['dataset_name'] in ['vocalsound', 'Nonspeech7k']:
            system_prompt = 'You are a vocal sound classification model.'
        elif meta['dataset_name'] == 'meld':
            system_prompt = 'You are a speech emotion recognition model.'
        elif meta['interactive'] == 'Audio-QA' or meta['audio_type'] == 'AudioEvent':
            # from: https://github.com/QwenLM/Qwen2.5-Omni/issues/178#issuecomment-2808125247
            system_prompt = 'You are a helpful assistant.'
        return system_prompt

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        if len(audio) == 1:
            audio = audio[0]

        task_prompt = self.get_prompt(msg)
        system_prompt = self.get_system_prompt(msg)

        if msg['meta']['interactive'] == 'Audio-analysis':
            messages = [
                {'role': 'system',
                    'content': [
                        {
                            'type': 'text',
                            'text': system_prompt
                        }
                    ]
                 },
                {'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': task_prompt
                        },
                        {
                            'type': 'audio',
                            'audio': audio
                        }
                    ]
                 },
            ]
        elif msg['meta']['interactive'] == 'Audio-QA':
            messages = [
                {'role': 'system',
                 'content': [
                     {
                         'type': 'text',
                         'text': system_prompt
                     }
                 ]
                 },
                {'role': 'user',
                 'content': [
                     {
                         'type': 'audio',
                         'audio': audio
                     }
                 ]
                 },
            ]
        else:
            raise NotImplementedError
        # only for dump
        prompt = system_prompt + '\n' + task_prompt
        print_once(f'Prompt: {prompt}')

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=True)
        assert audio is not None

        inputs = self.processor(text=text,
                                audio=audios,
                                images=images,
                                videos=videos,
                                return_tensors='pt',
                                padding=True, use_audio_in_video=True)

        inputs = inputs.to('cuda').to(self.model.dtype)

        if msg['meta']['task'] == 'ASR':
            # https://github.com/QwenLM/Qwen2.5-Omni/issues/79
            generated_ids = self.model.generate(**inputs, use_audio_in_video=True, return_audio=False,
                                                thinker_max_new_tokens=256, thinker_do_sample=False, repetition_penalty=1.0)

        else:
            generated_ids = self.model.generate(
                **inputs, use_audio_in_video=True, return_audio=False, thinker_do_sample=False)

        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        pred = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return prompt, pred
