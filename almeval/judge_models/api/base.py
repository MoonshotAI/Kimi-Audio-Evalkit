import asyncio
import copy as cp
import random as rd
import traceback
from abc import abstractmethod

from loguru import logger

from ...utils import parse_file


class BaseAPI:

    allowed_types = ['text', 'audio']
    allowed_roles = ['system', 'user', 'assistant']

    def __init__(self,
                 retry=10,
                 wait=3,
                 system_prompt=None,
                 verbose=False,
                 fail_msg='Failed to obtain answer via API.',
                 **kwargs):
        """Base Class for all APIs.

        Args:
            retry (int, optional): The retry times for `generate_inner`. Defaults to 10.
            wait (int, optional): The wait time after each failed retry of `generate_inner`. Defaults to 3.
            system_prompt (str, optional): Defaults to None.
            verbose (bool, optional): Defaults to True.
            fail_msg (str, optional): The message to return when failed to obtain answer.
                Defaults to 'Failed to obtain answer via API.'.
            **kwargs: Other kwargs for `generate_inner`.
        """

        self.wait = wait
        self.retry = retry
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fail_msg = fail_msg

        if len(kwargs):
            logger.info(f'BaseAPI received the following kwargs: {kwargs}')
            logger.info('Will try to use them as kwargs for `generate`. ')
        self.default_kwargs = kwargs

    @abstractmethod
    async def generate_inner(self, inputs, **kwargs):
        """The inner function to generate the answer.

        Returns:
            tuple(int, str, str): ret_code, response, log
        """
        raise NotImplementedError(
            'For APIBase, generate_inner is an abstract method. ')

    async def working(self):
        """If the API model is working, return True, else return False.

        Returns:
            bool: If the API model is working, return True, else return False.
        """
        retry = 3
        while retry > 0:
            ret = await self.generate('hello')
            if ret is not None and ret != '' and self.fail_msg not in ret:
                return True
            retry -= 1
        return False

    def check_content(self, msgs):
        """Check the content type of the input. Four types are allowed: str, dict, liststr, listdict.

        Args:
            msgs: Raw input messages.

        Returns:
            str: The message type.
        """
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    def preproc_content(self, inputs):
        """Convert the raw input messages to a list of dicts.

        Args:
            inputs: raw input messages.

        Returns:
            list(dict): The preprocessed input messages. Will return None if failed to preprocess the input.
        """
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            assert 'type' in inputs and 'value' in inputs
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime == 'unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                assert 'type' in item and 'value' in item
                mime, s = parse_file(item['value'])
                if mime is None:
                    assert (item['type'] == 'text')
                else:
                    assert mime.split('/')[0] == item['type']
                    item['value'] = s
            return inputs
        else:
            return None

    # May exceed the context windows size, so try with different turn numbers.
    async def chat_inner(self, inputs, **kwargs):
        _ = kwargs.pop('dataset', None)
        while len(inputs):
            try:
                return await self.generate_inner(inputs, **kwargs)
            except Exception:
                inputs = inputs[1:]
                while len(inputs) and inputs[0]['role'] != 'user':
                    inputs = inputs[1:]
                continue
        return -1, self.fail_msg + ': ' + 'Failed with all possible conversation turns.', None

    async def chat(self, messages, **kwargs1):
        """The main function for multi-turn chatting. Will call `chat_inner` with the preprocessed input messages."""
        assert hasattr(
            self, 'chat_inner'), 'The API model should has the `chat_inner` method. '
        for msg in messages:
            assert isinstance(
                msg, dict) and 'role' in msg and 'content' in msg, msg
            assert self.check_content(msg['content']) in [
                'str', 'dict', 'liststr', 'listdict'], msg
            msg['content'] = self.preproc_content(msg['content'])
        # merge kwargs
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)

        answer = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        await asyncio.sleep(T)

        assert messages[-1]['role'] == 'user'

        for i in range(self.retry):
            try:
                ret_code, answer, log = await self.chat_inner(messages, **kwargs)
                if ret_code == 0 and self.fail_msg not in answer and answer != '':
                    if self.verbose:
                        print(answer)
                    return answer
                elif self.verbose:
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except Exception:
                            logger.warning(
                                f'Failed to parse {log} as an http response. ')
                    logger.info(
                        f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
            except Exception as err:
                if self.verbose:
                    logger.error(f'An error occurred during try {i}:')
                    logger.error(err)
            # delay before each retry
            T = rd.random() * self.wait * 2
            await asyncio.sleep(T)

        return self.fail_msg if answer in ['', None] else answer

    async def generate(self, message, **kwargs1):
        """The main function to generate the answer. Will call `generate_inner` with the preprocessed input messages.

        Args:
            message: raw input messages.

        Returns:
            str: The generated answer of the Failed Message if failed to obtain answer.
        """
        assert self.check_content(message) in [
            'str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        message = self.preproc_content(message)
        assert message is not None and self.check_content(
            message) == 'listdict'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'

        # merge kwargs
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)

        answer = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        await asyncio.sleep(T)

        for i in range(self.retry):
            try:
                # The model from msh.py returns four results, while the other models return three results.
                _result_tuple = await self.generate_inner(message, **kwargs)
                if len(_result_tuple) == 3:
                    ret_code, answer, log = _result_tuple
                elif len(_result_tuple) == 4:
                    ret_code, answer, log, meta_info = _result_tuple
                elif len(_result_tuple) == 5:
                    ret_code, answer, log, meta_info, response_tokens = _result_tuple

                if ret_code == 0 and self.fail_msg not in answer and answer != '':
                    if self.verbose:
                        print(answer)
                    if len(_result_tuple) == 4:
                        return {'prediction': answer, 'meta_info': meta_info}
                    elif len(_result_tuple) == 5:
                        return {'prediction': answer, 'meta_info': meta_info, 'response_tokens': response_tokens}
                    else:
                        return answer
                elif self.verbose:
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except Exception:
                            logger.warning(
                                f'Failed to parse {log} as an http response. ')
                    logger.info(
                        f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
            except Exception as err:
                if self.verbose:
                    logger.error(f'An error occurred during try {i}:')
                    logger.error(err)
                    logger.error(traceback.format_exc())
                _result_tuple = []
            # delay before each retry
            T = rd.random() * self.wait * 2
            await asyncio.sleep(T)
        if len(_result_tuple) == 4:

            return {'prediction': self.fail_msg if answer in ['', None] else answer, 'meta_info': meta_info}
        elif len(_result_tuple) == 5:
            return {'prediction': self.fail_msg if answer in ['', None] else answer, 'meta_info': meta_info, 'response_tokens': response_tokens}
        else:
            print(_result_tuple)
            return {'prediction': self.fail_msg if answer in ['', None] else answer, 'response_tokens': 0}
