import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.misc import print_once
from .base import BaseModel
from .stepaudio.tokenizer import StepAudioTokenizer
from .stepaudio.utils import load_audio, load_optimus_ths_lib
from huggingface_hub import snapshot_download


class StepAudio(BaseModel):
    NAME = 'StepAudio'

    def __init__(self, model_path: str | None = None):
        super().__init__()
        # step-audio requires tokenizer & llm, if model_path is local path, try to find tokenizer & llm in the path
        # else, load from huggingface
        if model_path is not None:
            tokenizer_path = os.path.join(model_path, 'Step-Audio-Tokenizer')
            llm_path = os.path.join(model_path, 'Step-Audio-Chat')
        else:
            tokenizer_path = snapshot_download('stepfun-ai/Step-Audio-Tokenizer')
            llm_path = snapshot_download('stepfun-ai/Step-Audio-Chat')
            
        load_optimus_ths_lib(os.path.join(llm_path, 'lib'))
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_path, trust_remote_code=True
        )
        self.encoder = StepAudioTokenizer(tokenizer_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )

    def inference(self, messages: list):
        text_with_audio = self.apply_chat_template(messages)
        token_ids = self.llm_tokenizer.encode(
            text_with_audio, return_tensors='pt')
        token_ids = token_ids.to('cuda')
        outputs = self.llm.generate(
            token_ids, max_new_tokens=2048, temperature=0.7, top_p=0.9, do_sample=True
        )
        output_token_ids = outputs[:, token_ids.shape[-1]: -1].tolist()[0]
        output_text = self.llm_tokenizer.decode(output_token_ids)
        return output_text

    @staticmethod
    def get_prompt(msg: dict):
        # according to https://arxiv.org/pdf/2502.11946
        meta = msg['meta']
        if meta['task'] == 'ASR':
            prompt = '请记录下你所听到的语音内容。'

        # a general prompt for audio-qa
        elif meta['interactive'] == 'Audio-QA':
            prompt = '请回答音频中的问题。'
        elif meta['audio_type'] == 'AudioEvent':
            prompt = f'请听音频后回答如下问题： {msg["text"]} '
        else:
            prompt = msg['text']
        return prompt

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        if len(audio) == 1:
            audio = audio[0]
        prompt = self.get_prompt(msg)

        system_msg = {
            'role': 'system',
            'content': prompt
        }

        print_once(f'Prompt: {prompt}')
        x = [system_msg,
             {'role': 'user',
              'content': {'type': 'audio', 'audio': audio}}]
        text = self.inference(x)
        return prompt, text

    def encode_audio(self, audio: str | torch.Tensor, sr=None):
        if isinstance(audio, str):
            audio_wav, sr = load_audio(audio)
        else:
            assert sr is not None
            audio_wav = audio
        audio_tokens = self.encoder(audio_wav, sr)
        return audio_tokens

    def apply_chat_template(self, messages: list):
        text_with_audio = ''
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                role = 'human'
            if isinstance(content, str):
                text_with_audio += f'<|BOT|>{role}\n{content}<|EOT|>'
            elif isinstance(content, dict):
                if content['type'] == 'text':
                    text_with_audio += f"<|BOT|>{role}\n{content['text']}<|EOT|>"
                elif content['type'] == 'audio':
                    if isinstance(content['audio'], torch.Tensor):
                        assert 'audio_sr' in msg
                        audio_tokens = self.encode_audio(
                            content['audio'], msg['audio_sr'])
                    else:
                        audio_tokens = self.encode_audio(content['audio'])
                    text_with_audio += f'<|BOT|>{role}\n{audio_tokens}<|EOT|>'
            elif content is None:
                text_with_audio += f'<|BOT|>{role}\n'
            else:
                raise ValueError(f'Unsupported content type: {type(content)}')
        if not text_with_audio.endswith('<|BOT|>assistant\n'):
            text_with_audio += '<|BOT|>assistant\n'
        return text_with_audio
