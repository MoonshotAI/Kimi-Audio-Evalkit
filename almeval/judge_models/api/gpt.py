import json
import os
import sys
import traceback

import aiohttp
import numpy as np
from loguru import logger

from .base import BaseAPI

APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 600,
                 api_base: str = None,
                 max_tokens: int = 1024,
                 use_azure: bool = False,
                 ocr_provider: str = None,
                 **kwargs):

        self.model = model
        self.api_base = api_base
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure
        self.ocr_provider = ocr_provider
        if 'step-1v' in model:
            env_key = os.environ.get('STEPAI_API_KEY', '')
            if key is None:
                key = env_key
        else:
            if use_azure:
                env_key = os.environ.get('AZURE_OPENAI_API_KEY', None)
                assert env_key is not None, 'Please set the environment variable AZURE_OPENAI_API_KEY. '

                if key is None:
                    key = env_key
                assert isinstance(key, str), (
                    'Please set the environment variable AZURE_OPENAI_API_KEY to your openai key. '
                )
            else:
                env_key = os.environ.get('OPENAI_API_KEY', '')
                if key is None:
                    key = env_key
                assert isinstance(key, str), (
                    f'Illegal openai_key {key}. '
                    'Please set the environment variable OPENAI_API_KEY to your openai key. '
                )

        self.key = key
        self.timeout = timeout

        super().__init__(wait=wait, retry=retry,
                         system_prompt=system_prompt, verbose=verbose, **kwargs)

        if use_azure:
            api_base_template = (
                '{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}'
            )
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', None)
            assert endpoint is not None, 'Please set the environment variable AZURE_OPENAI_ENDPOINT. '
            deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', None)
            assert deployment_name is not None, 'Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME. '
            api_version = os.getenv('OPENAI_API_VERSION', None)
            assert api_version is not None, 'Please set the environment variable OPENAI_API_VERSION. '

            self.api_base = api_base_template.format(
                endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                api_version=os.getenv('OPENAI_API_VERSION')
            )
        else:
            if api_base is None:
                if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
                    logger.error(
                        'Environment variable OPENAI_API_BASE is set. Will use it as api_base. ')
                    api_base = os.environ['OPENAI_API_BASE']
                else:
                    api_base = 'OFFICIAL'

            assert api_base is not None

            if api_base in APIBASES:
                self.api_base = APIBASES[api_base]
            elif api_base.startswith('http'):
                self.api_base = api_base
            else:
                logger.error('Unknown API Base. ')
                sys.exit(-1)

    def prepare_itlist(self, inputs):
        assert all([x['type'] == 'text' for x in inputs])
        text = '\n'.join([x['value'] for x in inputs])
        content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(
            ['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(
                    dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(
                dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    async def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        context_window = GPT_context_window(self.model)
        max_tokens = min(max_tokens, context_window -
                         self.get_token_len(inputs))
        if 0 < max_tokens <= 100:
            logger.warning(
                'Less than 100 tokens left, '
                'may exceed the context window with some additional meta symbols. '
            )
        if max_tokens <= 0:
            return 0, self.fail_msg + 'Input string longer than context window. ', 'Length Exceeded. '

        # Will send request if use Azure, dk how to use openai client for it
        if self.use_azure:
            headers = {'Content-Type': 'application/json', 'api-key': self.key}
        else:
            headers = {'Content-Type': 'application/json',
                       'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            **kwargs)
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                self.api_base,
                headers=headers, json=payload, timeout=self.timeout * 1.1)
            ret_code = response.status
            ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
            answer = self.fail_msg
            response_text = await response.text()
        try:
            resp_struct = json.loads(response_text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f'Error: {e}')
            print(f'Traceback:\n{traceback.format_exc()}')
            print(f'Response:\n{response_text}')
            answer = self.fail_msg
            # import pdb; pdb.set_trace()
        return ret_code, answer, response

    def get_token_len(self, inputs) -> int:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception:
            enc = tiktoken.encoding_for_model('gpt-4')
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if 'role' in item:
                tot += self.get_token_len(item['content'])
            elif item['type'] == 'text':
                tot += len(enc.encode(item['value']))
        return tot
