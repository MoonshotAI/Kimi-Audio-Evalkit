import re

import pandas as pd
from loguru import logger

from ..judge_models import get_judge_model
from .base import AudioBaseDataset

REF_QA_PRPMPT = """
You are an expert in judging answer correctness. If the model's output is correct, output "yes", otherwise output "no".
You need to explain your judgment process first, then output "yes" or "no".

[Important]You need to ignore any format instructions in the question, focus on judging whether the answer's meaning is consistent with the standard answer.

- The language is not important, if the model's output and standard answer are in different languages but have the same meaning, judge as "yes"


The input format is:
Input:
Question: The question from user
Model Answer: The answer from models
Ground Truth Answer: The ground truth answer
Explanation: The explanation of your judgment process

Example 1:
Input:
Question: Based on the given audio, identify the source of the speaking voice.
Model Answer: A man is speaking in the audio.
Ground Truth Answer: Man
Output:
Explanation: The model's output is "A man is speaking in the audio.", this is a detail description of the ground truth answer "Man". So the model's output is correct.
Result: yes


Task:
Input:
Question: {question}
Model Answer: {prediction}
Ground Truth Answer: {answer}
Output:
"""

SQ_QA_PROMPT = """
### Question
{question}

### Reference answer
{answer}

### Candidate answer
{prediction}

Is the candidate answer correct based on the question and reference answer?
Please only output a single "yes" or "no". Do not output anything else.
"""


OpenAudioBench_PROMPTS = {
    'alpaca_eval': """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user \
question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth,\
creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as \
objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly \
following this format: "[[rating]]", for example: "Rating: [[5]]".
[Question]
{question}

[The Start of Assistant’s Answer]
{prediction}
[The End of Assistant’s Answer]""",

    'llama_questions': """
## Background
You are a professional QA evaluation expert. You need to assess whether the model's answer is correct based on the \
standard answer.\n\n
## Scoring Criteria
Correct: The answer matches or is equivalent to the standard answer \n
Incorrect: The answer is wrong or irrelevant to the question \n\n
## Evaluation Guidelines
1. The expression of answers can be flexible, not requiring exact matches. For example: \n
   - Numbers can be expressed in either Arabic numerals or words \n
   - Proper nouns can be in either English or Chinese \n
   - Differences in punctuation can be ignored \n
2. Focus on whether the core meaning of the answer is correct \n
## Output Format
Provide the reasoning for your score, then generate the result in "[]" format and make sure it contains "the score is \
[Correct]" or "the score is [Incorrect]", for example:
```
The answer is correct and equivalent to the standard answer, the score is [Correct]
```
or
```
The answer is incorrect and does not match the standard answer, the score is [Incorrect]
```
\n\n
## Question:
{question}
## Standard Answer:
{answer}
## Model's Answer:
{prediction}
""",
    'reasoning_qa': """
## 背景
现在你是一个大学数学老师。你需要依据 标准答案 来判断每道题的得分\n\n
## 判分依据
5分答案：满分答案，需要回答的答案正确，同时过程正确，且回答考虑到了各种可能性，考虑全面 \n
4分答案：答案正确，但是没有过程 \n
3分答案：答案错误，过程大部分正确；或者答案正确，但是过程出现明显错误 \n
2分答案：答案错误，且过程大部分错误 \n
1分答案：答案错误，过程和思路全错\n\n
## 其他注意事项
你需要忽略格式问题，以下都是一些等价的情况，不应该作为答案正确性的判断，比如 \n
1）latex格式表达的公式，普通格式表达的公式 \n
2）分数和小数表达的数值：比如1/3和0.33都算对 \n
3）关于π的表达：比如π、pi、3.14都是等价的 \n
4）关于常数的表达：比如n、k等常数表达都是等价的 \n
等，还有很多其他类似的等价表达 \n\n
## 生成格式
写出判分理由，再以"[]"的格式生成分数，比如：
```
这道题回答正确，但是没有中间过程，因此得4分，得分是[4]
```
\n\n
## 题目
{question}
## 标准答案:
{answer}
## 学生回答:
{prediction}
""",

    'trivia_qa': """
Your will be given a question, the reference answers to that question, and an answer to be judged. Your tasks is to \
judge whether the answer to be judged is correct, given the question and reference answers. An answer considered \
correct expresses or contains the same meaning as at least **one of** the reference answers. The format and the tone \
of the response does not matter.\
You should respond in JSON format. First provide a one-sentence concise analysis for the judgement in field ‘analysis‘,\
 then your judgment in field ‘judgment‘. For example,\
'''json
{{"analysis": "<a one-sentence concise analysis for the judgement>", "judgment": < your final judgment, "correct" \
or "incorrect">}}
'''
# Question
{question}
# Reference Answer
{answer}
# Answer To Be Judged
{prediction}
""",

    'web_questions': """
Your will be given a question, the reference answers to that question, and an answer to be judged. Your tasks is to \
judge whether the answer to be judged is correct, given the question and reference answers. An answer considered \
correct expresses or contains the same meaning as at least **one of** the reference answers. The format and the tone \
of the response does not matter.\
You should respond in JSON format. First provide a one-sentence concise analysis for the judgement in field ‘analysis‘,\
 then your judgment in field ‘judgment‘. For example,\
'''json
{{"analysis": "<a one-sentence concise analysis for the judgement>", "judgment": < your final judgment, "correct" \
or "incorrect">}}
'''
# Question
{question}
# Reference Answer
{answer}
# Answer To Be Judged
{prediction}
"""
}


class AudioRefQADataset(AudioBaseDataset):
    INTERACTIVE = 'Audio-analysis'
    TASK = 'Ref-QA'
    LANG = None

    def evaluate(self, eval_file, dump_judge=True, method='default'):
        if method == 'vb-qa':
            from qa_metrics.pedant import PEDANT
            self.pedant = PEDANT()
            metrics, judge_results = self.evaluate_vb_qa(eval_file)
            judge_model_name = 'vb-qa'
        else:
            if method == 'default':
                method = 'gpt-4o-mini'
            judge_model = get_judge_model(method)
            metrics, judge_results = self.evaluate_llm(
                eval_file, judge_model)
            judge_model_name = judge_model.model

        model_name = self.get_model_name(eval_file)
        result = self.format_performance(
            model_name, metrics, eval_method=method)

        if dump_judge:
            # dump the judge result to the eval_file
            all_df = []
            for task, judge_result in judge_results.items():
                df = pd.DataFrame(judge_result)
                all_df.append(df)
            all_df = pd.concat(all_df)
            save_file = eval_file.replace(
                '.jsonl', f'_{judge_model_name}_judge.jsonl')
            all_df.to_json(save_file, orient='records',
                           lines=True, force_ascii=False)
        return result

    def evaluate_vb_qa(self, eval_file):
        df = pd.read_json(eval_file, lines=True)
        metrics = {}
        judge_results = {}
        for task, group in df.groupby('subset'):
            if task == 'sentiment':
                continue
            ground_truth = group['answer'].astype(str).to_list()
            preds = group['prediction'].astype(str).to_list()
            prompts = group['prompt'].astype(str).to_list()
            results = []
            for idx in range(len(preds)):
                pred = preds[idx]
                gt = ground_truth[idx]
                prompt = prompts[idx]
                score = self.pedant.evaluate([gt], pred, prompt)
                results.append((idx, 'yes' if score is True else 'no'))

            task_result, judge_result = self.collect_acc(results, group)
            metrics[task] = task_result
            print(f'{task} result: {task_result}')
            judge_results[task] = judge_result
        return metrics, judge_results

    def evaluate_llm(self, eval_file, judge_model=None):
        df = pd.read_json(eval_file, lines=True)
        metrics = {}
        judge_results = {}
        if self.DATASET_NAME == 'sd-qa':
            temperature = 0.0
            top_p = 0.95
            prompt = SQ_QA_PROMPT
        else:
            temperature = 0.0
            top_p = 0.95
            prompt = REF_QA_PRPMPT
        for task, group in df.groupby('subset'):
            # audio QA时，question是prompt，因为没有对于Audio的question
            question = self.get_LLM_query(group)
            gt = group['answer'].astype(str).to_list()
            pred = group['prediction'].astype(str).to_list()
            results = self.run_llm_judge(
                judge_model, prompt, pred=pred, gt=gt, question=question, temperature=temperature, top_p=top_p)
            task_result, judge_result = self.collect_acc(results, group)
            metrics[task] = task_result
            print(f'{task} result: {task_result}')
            judge_results[task] = judge_result
        return metrics, judge_results


class ClothoAQA(AudioRefQADataset):
    DATASET_NAME = 'ClothoAQA'
    DATASET_SERIES = 'ClothoAQA'
    AUDIO_TYPE = 'AudioEvent'


class SDQA(AudioRefQADataset):
    DATASET_NAME = 'sd-qa'
    DATASET_SERIES = 'VoiceBench'
    AUDIO_TYPE = 'Speech'
    INTERACTIVE = 'Audio-QA'


class OpenAudioBench(AudioRefQADataset):
    DATASET_NAME = 'OpenAudioBench'
    DATASET_SERIES = 'OpenAudioBench'
    AUDIO_TYPE = 'Speech'
    INTERACTIVE = 'Audio-QA'

    def evaluate_llm(self, eval_file, judge_model=None):
        """follow code from: https://huggingface.co/datasets/baichuan-inc/OpenAudioBench/tree/main
           and : https://arxiv.org/pdf/2502.17239
        """
        df = pd.read_json(eval_file, lines=True)

        # Evaluate the model performance
        metrics = {}
        judge_results = {}
        for subset, group in df.groupby('subset'):
            logger.info(f'evaluating {subset}...')
            question = self.get_LLM_query(group)
            gt = group['answer'].astype(str).to_list()
            pred = group['prediction'].astype(str).to_list()

            QUERY_PROMPT = OpenAudioBench_PROMPTS[subset]
            if subset == 'alpaca_eval':
                results = self.run_llm_judge(
                    judge_model, QUERY_PROMPT, pred=pred, question=question)
            else:
                results = self.run_llm_judge(
                    judge_model, QUERY_PROMPT, pred=pred, gt=gt, question=question)
            judge_result = []
            for i, res in results:
                org_item = group.iloc[i].to_dict()
                org_item['judge_result'] = res
                judge_result.append(org_item)
                if i == 0:
                    print(
                        f'{subset} question: {question[0]},  pred: {pred[0]}, gt: {gt[0]}, judge result: {res}')
            judge_results[subset] = judge_result

            if subset == 'alpaca_eval':
                cnt = 0
                sum_score = 0
                for i, res in results:
                    match = re.search(r'\[\[(\d+)\]\]', res)
                    if match:
                        score = int(match.group(1))
                        assert 1 <= score <= 10
                        sum_score += (score*10)
                        cnt += 1
                task_result = {
                    'score': round(sum_score / cnt, 2),
                    'total': len(pred),
                    'invalid': len(pred) - cnt

                }
                metrics[subset] = task_result

            elif subset == 'reasoning_qa':
                cnt = 0
                sum_score = 0
                for i, res in results:
                    scores = re.findall(r'\[([0-5])\]', res)
                    if len(scores) >= 1:
                        sum_score += (int(scores[-1])*20)
                        cnt += 1
                task_result = {
                    'score': round(sum_score / cnt, 2),
                    'total': len(pred),
                    'invalid': len(pred) - cnt
                }
                metrics[subset] = task_result
            elif subset == 'llama_questions':
                correct_cnt = 0
                cnt = 0
                for i, res in results:
                    try:
                        score = re.findall(
                            r'[Tt]he score is \[(Correct|Incorrect|correct|incorrect)\]', res)[0]
                        if score.lower() == 'correct':
                            correct_cnt += 1
                        cnt += 1
                    except Exception as e:
                        print(f'Error: {res}, reasons: {e}')
                        continue
                task_result = {
                    'acc': round((correct_cnt / cnt)*100, 2),
                    'total': len(pred),
                    'invalid': len(pred) - cnt
                }
                metrics[subset] = task_result

            else:
                correct_cnt = 0
                cnt = 0
                for i, res in results:
                    res = res[res.find('{'):res.find('}')+1]

                    try:
                        eval_js = eval(res)
                        if eval_js['judgment'].lower() == 'correct':
                            correct_cnt += 1
                        cnt += 1
                    except Exception as e:
                        print(f'Error: {res}, reasons: {e}')
                        continue

                task_result = {
                    'acc': round((correct_cnt / cnt)*100, 2),
                    'total': len(pred),
                    'invalid': len(pred) - cnt
                }
                metrics[subset] = task_result

            print(f'{subset} performance: {metrics}')

        return metrics, judge_results
