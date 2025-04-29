import json
import re
import types
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.misc import print_once
from .base import BaseModel
from .patch import patch_baichuan_load_audio_waveform

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.set_num_threads(1)


class BaichuanAudioBase(BaseModel):
    NAME = 'Baichuan-Audio'

    def __init__(self, model_path='baichuan-inc/Baichuan-Audio-Base', **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, model_max_length=128000)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='cuda',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.model.config.use_cache = True
        self.model.bind_processor(
            self.tokenizer, training=False, relative_path='/')

        audio_processor = self.model.processor.audio_processor
        audio_processor.load_audio_waveform = types.MethodType(
            patch_baichuan_load_audio_waveform, audio_processor)

        self.audio_start_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_start_token_id)
        self.audio_end_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_end_token_id)
        torch.cuda.empty_cache()

    def get_prompt(self, msg: dict):
        meta = msg['meta']
        if meta['task'] == 'ASR':
            # from: https://github.com/baichuan-inc/Baichuan-Audio/blob/805d456433dbf3e0edb2bdd302f733a4bd38ea84/web_demo/base_asr_demo.py#L84C19-L84C27
            prompt = '将语音转录为文本:'
        # to invoke basemodel continuous output
        elif meta['interactive'] == 'Audio-QA':
            prompt = '对音频中的问题，你的回答是:'
        elif meta['audio_type'] == 'AudioEvent':
            prompt = f'请听音频后回答如下问题： {msg["text"]} 你的回答是: '
        else:
            prompt = msg['text']
            # to invoke basemodel continuous output
            end_punctuation = ['.', '?', '!', '。', '？', '！']
            if prompt.endswith(tuple(end_punctuation)):
                prompt = prompt + ' Your answer to this question is:'
            else:
                prompt = prompt + ' . ' + 'Your answer to this question is:'
        return prompt

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        if len(audio) == 1:
            audio = audio[0]
        audio_tokens_all = []
        prompt = self.get_prompt(msg)
        audio_tokens = self.audio_start_token + \
            json.dumps({'path': audio}) + self.audio_end_token
        audio_tokens_all.append(audio_tokens)

        print_once(f'Prompt: {prompt}')
        prompt = prompt + ''.join(audio_tokens_all)

        ret = self.model.processor([prompt])
        ret_audios = ret.audios.cuda() if ret.audios is not None else None
        ret_encoder_length = ret.encoder_length.cuda(
        ) if ret.encoder_length is not None else None
        ret_bridge_length = ret.bridge_length.cuda(
        ) if ret.bridge_length is not None else None
        predicted_ids = self.model.generate(input_ids=ret.input_ids.cuda(),
                                            attention_mask=ret.attention_mask.cuda(),
                                            labels=None,
                                            audios=ret_audios,
                                            encoder_length=ret_encoder_length,
                                            bridge_length=ret_bridge_length,
                                            max_new_tokens=700,
                                            num_beams=1,
                                            do_sample=False,
                                            num_return_sequences=1,
                                            repetition_penalty=1.3)
        generated = self.tokenizer.batch_decode(
            predicted_ids[:, ret.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return prompt, generated


class BaichuanAudioChat(BaichuanAudioBase):
    """Chat model for Audio-QA only, no text prompt given.
    """
    role_prefix = {
        'system': '<B_SYS>',
        'user': '<C_Q>',
        'assistant': '<C_A>',
        'audiogen': '<audiotext_start_baichuan>'
    }
    special_token_partten = re.compile(
        r'<\|endoftext\|>|<audiogen_start_baichuan>|<audiogen_end_baichuan>')
    sample_rate = 24000
    NAME = 'Baichuan-Audio-Chat'

    def __init__(self, model_path='baichuan-inc/Baichuan-Audio-Instruct', **kwargs):
        super().__init__(model_path, **kwargs)

    def preprocess_messages(self, messages):
        text = ''
        for i, msg in enumerate(messages):
            text += self.role_prefix[msg['role']]
            text += msg['content']
        text += self.role_prefix['assistant']
        return text

    def generate_text_step(self, pret, plen, kv_cache_flag):
        if not kv_cache_flag:
            textret = self.model.generate(
                pret.input_ids.cuda(),
                attention_mask=pret.attention_mask.cuda(),
                audios=pret.audios.cuda() if pret.audios is not None else None,
                encoder_length=pret.encoder_length.cuda(
                ) if pret.encoder_length is not None else None,
                bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
                tokenizer=self.tokenizer,
                max_new_tokens=1024,
                stop_strings=['<|endoftext|>'],
                do_sample=True, temperature=0.8, top_k=20, top_p=0.85, repetition_penalty=1.1, return_dict_in_generate=True,
            )
        else:
            textret = self.model.generate(
                pret.sequences,
                attention_mask=torch.ones_like(pret.sequences),
                tokenizer=self.tokenizer,
                past_key_values=(pret.past_key_values),
                stop_strings=[self.audiogen_start_token,
                              ',', '!', '?', '，', '。', '！', '？', '. '],
                max_new_tokens=50, do_sample=True, temperature=0.3, top_k=20, top_p=0.85, repetition_penalty=1.05, return_dict_in_generate=True,
            )
        newtext = self.tokenizer.decode(textret.sequences[0, plen:])
        return textret, newtext

    def generate_response(self, content):
        pret = self.model.processor([content])
        plen = pret.input_ids.shape[1]
        _, text_segment = self.generate_text_step(pret, plen, False)
        full_text = re.sub(self.special_token_partten, '', text_segment)
        show_text = re.sub(self.special_token_partten, '', text_segment)
        yield show_text, full_text, None

    def get_prompt(self, msg: dict):
        # according to https://github.com/baichuan-inc/Baichuan-Audio/blob/main/web_demo/base_asr_demo.py
        meta = msg['meta']
        if meta['task'] == 'ASR':
            # from: https://github.com/baichuan-inc/Baichuan-Audio/blob/805d456433dbf3e0edb2bdd302f733a4bd38ea84/web_demo/base_asr_demo.py#L84C19-L84C27
            prompt = '将语音转录为文本。'

        elif meta['interactive'] == 'Audio-QA':
            # from: https://github.com/baichuan-inc/Baichuan-Audio/blob/805d456433dbf3e0edb2bdd302f733a4bd38ea84/web_demo/s2s_gradio_demo_cosy_multiturn.py#L309
            prompt = '请用【邻家女声】这个声音回答问题。'

        elif meta['audio_type'] == 'AudioEvent':
            prompt = f'请听音频后回答如下问题： {msg["text"]}'
        else:
            prompt = msg['text']
        return prompt

    def generate_inner(self, msg: dict):
        audio = msg['audio']

        if len(audio) == 1:
            audio = audio[0]

        sys_prompt = self.get_prompt(msg)
        audio_tokens_all = []
        audio_tokens = self.audio_start_token + \
            json.dumps({'path': audio}) + self.audio_end_token
        audio_tokens_all.append(audio_tokens)
        prompt = ''.join(audio_tokens_all)

        msgs = [
            {
                'role': 'system',
                'content': sys_prompt
            },
            {
                'role': 'user',
                'content': prompt,
            }
        ]
        message = self.preprocess_messages(msgs)
        for _, full_text, _ in self.generate_response(message):
            pass
        return sys_prompt, full_text
