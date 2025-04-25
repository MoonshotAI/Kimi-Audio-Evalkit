import os

from .api import OpenAIWrapper


async def judge_response(prompt: str, judge_model: OpenAIWrapper,
                         temperature=0.0,
                         top_p=0.95,
                         max_tokens=1024,
                         frequency_penalty=0,
                         presence_penalty=0,
                         stop=None):
    judge_result_str = ''
    try:
        judge_result_str = await judge_model.generate(prompt,
                                                      temperature=temperature,
                                                      top_p=top_p,
                                                      max_tokens=max_tokens,
                                                      frequency_penalty=frequency_penalty,
                                                      presence_penalty=presence_penalty, stop=stop)
        judge_result_str = judge_result_str.strip()
        return judge_result_str

    except Exception as e:
        print(f'Judge failed!: {e}')
        print(f'Output:\n{judge_result_str}')
        return None


def get_gpt4o_model():
    cur_judge_kwargs = {}
    cur_judge_kwargs['model'] = 'gpt-4o'
    cur_judge_kwargs['api_base'] = 'https://api.openai.com/v1/chat/completions'
    assert os.environ['OPENAI_API_KEY']

    judge_model = OpenAIWrapper(**cur_judge_kwargs)
    return judge_model


def get_gpt4o_mini_model():
    cur_judge_kwargs = {}
    cur_judge_kwargs['model'] = 'gpt-4o-mini'
    cur_judge_kwargs['api_base'] = 'https://api.openai.com/v1/chat/completions'
    cur_judge_kwargs['system_prompt'] = "You are a helpful assistant who tries to help answer the user's question."
    assert os.environ['OPENAI_API_KEY']

    judge_model = OpenAIWrapper(**cur_judge_kwargs)
    return judge_model


def get_judge_model(method='default'):
    if method == 'default':
        method = 'gpt-4o-mini'
    elif method == 'gpt-4o':
        return get_gpt4o_model()
    elif method == 'gpt-4o-mini':
        return get_gpt4o_mini_model()
    else:
        raise ValueError(f'Unsupported evaluation method: {method}')
