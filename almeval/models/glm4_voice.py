from io import BytesIO

import librosa
import numpy as np
import torch
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          WhisperFeatureExtractor)

from ..utils.misc import print_once
from .base import BaseModel
from .glm4voice.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from .glm4voice.speech_tokenizer.utils import extract_speech_token
from .patch import (patch_chatglm_model_init,
                    patch_glm4_voice_update_model_kwargs_for_generation)


class GLM4Voice(BaseModel):
    NAME = 'GLM4-Voice'

    def __init__(self, model_path='THUDM/glm-4-voice-9b',
                 device='cuda',
                 **kwargs):
        self.device = device
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_class = AutoModel.from_config(config, trust_remote_code=True)
        ChatGLMModel = AutoModel.from_config(
            model_class.config).transformer.__class__
        original_init = ChatGLMModel.__init__
        ChatGLMModel.__init__ = patch_chatglm_model_init(original_init)

        self.glm_model = model_class.from_pretrained(
            model_path,
            config=config,
            device=device,
            trust_remote_code=True
        ).to(device).eval()
        self.glm_model._update_model_kwargs_for_generation = patch_glm4_voice_update_model_kwargs_for_generation
        self.audio_tokenizer = Glm4Tokenizer()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        self.DEFAULT_SYSTEM_PROMPT = ('<|system|>\nUser will provide you with a speech instruction. Do it step by step. First,'
                                      'think about the instruction and respond in a interleaved manner,'
                                      'with 13 text token followed by 26 audio tokens. ')
        torch.cuda.empty_cache()

    def get_token_ids(self, audio: BytesIO):

        audio_tokens = self.audio_tokenizer.tokenize(audio)[0]
        audio_tokens = ''.join([f'<|audio_{x}|>' for x in audio_tokens])
        audio_tokens = '<|begin_of_audio|>' + audio_tokens + '<|end_of_audio|>'
        return audio_tokens

    def get_prompt(self, msg: dict):

        meta = msg['meta']
        prefix = '<|system|>\n'
        prompt = ''
        if meta['task'] == 'ASR':
            # from: https://arxiv.org/pdf/2502.11946
            prompt = prefix + '请写下你听到的语音内容。'
            prompt += ' listen to the audio carefully and respond with only text tokens'
        elif meta['interactive'] == 'Audio-QA':
            # from:https://github.com/THUDM/GLM-4-Voice/blob/eb00ce9142e8d98b0ed7c57cd47e0d6d5dce9a1a/web_demo.py#L91
            prompt = self.DEFAULT_SYSTEM_PROMPT
        elif meta['audio_type'] == 'AudioEvent':
            prompt = f'请听音频后回答如下问题： {msg["text"]} .'
        else:
            # all other audio-analysis tasks.
            prompt = prefix + msg['text']
            # help to output text tokens.
            prompt += ' listen to the audio carefully and respond with only text tokens'
        return prompt

    def _generate(self, sysmtem_prompt, audio, meta, temperature=0.2, top_p=0.8, max_new_tokens=2048):

        user_input = self.get_token_ids(audio)

        inputs = sysmtem_prompt
        inputs += f'<|user|>\n{user_input}<|assistant|>streaming_transcription\n'

        print_once(f'Prompt: {sysmtem_prompt}')

        inputs = self.glm_tokenizer([inputs], return_tensors='pt')
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            out_tokens = self.glm_model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p)
            )

        text_tokens = []
        audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        end_token_id = self.glm_tokenizer.convert_tokens_to_ids('<|user|>')

        out_tokens = out_tokens.squeeze(0).tolist()
        # out_tokens需要去掉prompt的部分
        out_tokens = out_tokens[len(inputs['input_ids'][0]):]

        for token_id in out_tokens:
            # logger.info(f'token_id: {token_id}, audio_offset: {audio_offset}, end_token_id: {end_token_id}')
            if token_id == end_token_id:
                break
            else:
                if token_id < audio_offset:
                    text_tokens.append(token_id)

        out_text = self.glm_tokenizer.decode(
            text_tokens, spaces_between_special_tokens=False)
        return out_text

    def generate_inner(self, msg: dict):
        audio = msg['audio']
        if len(audio) == 1:
            audio = audio[0]
        sysmtem_prompt = self.get_prompt(msg)
        return sysmtem_prompt, self._generate(sysmtem_prompt, audio, msg)


class Glm4Tokenizer:
    def __init__(self, tokenizer_path='THUDM/glm-4-voice-tokenizer',
                 device: str = 'cuda',):
        self.whisper_model = WhisperVQEncoder.from_pretrained(
            tokenizer_path).eval().to(device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            tokenizer_path)
        self.device = device
        self.in_sr = 16000
        self.out_sr = 22050

    def tokenize(self, audio: np.ndarray | torch.Tensor | str | BytesIO, sr: int | None = None) -> torch.Tensor:
        """Tokenize audio using tokenizer
        Args:
            audio: audio data or path, np.ndarray or torch.Tensor or str.
                   if audio is np.ndarray or torch.Tensor, sr must be provided.
                   if audio is str and sr is not None, will resample audio to sr.
            sr: sample rate of audio signal, by default None. if None, will use self.in_sr
        Returns:
            audio_tokens: torch.Tensor, shape (B, T), B is batch size, T is token length.
        """
        in_sr = sr or self.in_sr
        if isinstance(audio, str) or isinstance(audio, BytesIO):
            audio, sr = librosa.load(audio, sr=in_sr)
            audio = torch.tensor(audio).unsqueeze(0)
            audio_info = (audio, sr)
        elif isinstance(audio, np.ndarray):
            assert sr is not None
            audio = torch.tensor(audio).unsqueeze(0)
            audio_info = (audio, sr)
        else:
            raise ValueError(
                f'audio must be np.ndarray or torch.Tensor or str or BytesIO, got {type(audio)}')
        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [audio_info]
        )[0]
        audio_tokens = torch.tensor(audio_tokens)
        if len(audio_tokens.shape) == 1:
            audio_tokens = audio_tokens.unsqueeze(0)
        return audio_tokens
