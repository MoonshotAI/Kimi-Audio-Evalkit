import random

import pandas as pd

from ..judge_models import get_judge_model
from .base import AudioBaseDataset

SINGLE_CHOICE_PRPMPT = """
You are an expert in judging answer correctness. If the model's output is correct, output "yes", otherwise output "no".
You need to explain your judgment process first, then output "yes" or "no".

[Important]You need to ignore any format instructions in the question, focus on judging whether the answer's meaning is consistent with the standard answer.

The model's output may not exactly match the text of any option. You need to understand the meaning of the model's output answer, rather than performing literal matching.

- If the model's output and standard answer are in different languages but have the same meaning, judge as "yes"
- If the model only outputs the option letter of the standard answer, judge as "yes"
- The format of the option letter is not important. For example, if the standard answer is A, and the model outputs (A), (a), A, or a, all should be judged as "yes". You should always interpret a single letter result as the option letter of the standard answer.
- If the model's output and the standard answer are synonyms, judge as "yes". For example, "happy" and "joy" are synonyms, "sad" and "sadness" are synonyms, "angry" and "anger" are synonyms, "surprised" and "surprise" are synonyms.

The input format is:
Input:
Question: The question from user
Model Answer: The answer from models
Ground Truth Answer: The ground truth answer
Explanation: The explanation of your judgment process

Example 1:
Input:
Question: Identify the predominant emotion in this speech.\nOptions:\n(A) neutral\n(B) joy\n(C) sadness\n(D) anger\n(E) surprise\n(F) fear\n(G) disgust\n.Answer with the option's letter from the given choices directly and only give the best option.
Model Answer: happy
Ground Truth Answer: joy
Output:
Explanation: The model's output is "happy", which is a synonym of "joy". The instruction of "Answer with the option's letter from the given choices directly and only give the best option." is ignored.
Result: yes

Task:
Input:
Question: {question}
Model Answer: {prediction}
Ground Truth Answer: {answer}
Output:
"""  # noqa


TUT2017_SINGLE_CHOICE_PRPMPT = """
You are an expert in judging answer correctness. If the model's output is correct, output "yes", otherwise output "no".
You need to explain your judgment process first, then output "yes" or "no".

[Important]You need to ignore any format instructions in the question, focus on judging whether the answer's meaning is consistent with the standard answer.

The model's output may not exactly match the text of any option. You need to understand the meaning of the model's output answer, rather than performing literal matching.

- If the model's output and standard answer are in different languages but have the same meaning, judge as "yes"
- If the model only outputs the option letter of the standard answer, judge as "yes"
- The format of the option letter is not important. For example, if the standard answer is A, and the model outputs (A), (a), A, or a, all should be judged as "yes". You should always interpret a single letter result as the option letter of the standard answer.
- If the model's output and the standard answer are synonyms, judge as "yes". For example, "happy" and "joy" are synonyms, "sad" and "sadness" are synonyms, "angry" and "anger" are synonyms, "surprised" and "surprise" are synonyms.

The input format is:
Input:
Question: The question from user
Model Answer: The answer from models
Ground Truth Answer: The ground truth answer

Example 1:
Input:
Question: Identify the acoustic scene in the audio.\nOptions:\n(A) beach\n(B) bus\n(C) cafe or restaurant\n(D) car\n(E) city center\n(F) forest path\n(G) grocery store\n(H) home\n(I) library\n(J) metro station\n(K) office\n(L) park\n(M) residential area\n(N) train\n(O) tram\n.Answer with the option's letter from the given choices directly and only give the best option.
Model Answer: O
Ground Truth Answer: residential_area
Output:
Result: no

Example 2:
Input:
Question: Identify the acoustic scene in the audio.\nOptions:\n(A) beach\n(B) bus\n(C) cafe or restaurant\n(D) car\n(E) city center\n(F) forest path\n(G) grocery store\n(H) home\n(I) library\n(J) metro station\n(K) office\n(L) park\n(M) residential area\n(N) train\n(O) tram\n.Answer with the option's letter from the given choices directly and only give the best option.
Model Answer: B
Ground Truth Answer: bus
Output:
Result: yes

Task:
Input:
Question: {question}
Model Answer: {prediction}
Ground Truth Answer: {answer}
Output:
"""  # noqa


COCHLSCENE_SINGLE_CHOICE_PRPMPT = """
You are an expert in judging answer correctness. If the model's output is correct, output "yes", otherwise output "no".
You need to explain your judgment process first, then output "yes" or "no".

[Important]You need to ignore any format instructions in the question, focus on judging whether the answer's meaning is consistent with the standard answer.

The model's output may not exactly match the text of any option. You need to understand the meaning of the model's output answer, rather than performing literal matching.

- If the model's output and standard answer are in different languages but have the same meaning, judge as "yes"
- If the model only outputs the option letter of the standard answer, judge as "yes"
- The format of the option letter is not important. For example, if the standard answer is A, and the model outputs (A), (a), A, or a, all should be judged as "yes". You should always interpret a single letter result as the option letter of the standard answer.
- If the model's output and the standard answer are synonyms, judge as "yes". For example, "happy" and "joy" are synonyms, "sad" and "sadness" are synonyms, "angry" and "anger" are synonyms, "surprised" and "surprise" are synonyms.

The input format is:
Input:
Question: The question from user
Model Answer: The answer from models
Ground Truth Answer: The ground truth answer

Example 1:
Input:
Question: Identify the acoustic scene in the audio.\nOptions:\n(A) bus\n(B) cafe\n(C) car\n(D) crowdedindoor\n(E) elevator\n(F) kitchen\n(G) park\n(H) residentialarea\n(I) restaurant\n(J) restroom\n(K) street\n(L) subway\n(M) subwaystation\n.Answer with the option's letter from the given choices directly and only give the best option.
Model Answer: A
Ground Truth Answer: Subway
Output:
Result: no

Example 2:
Input:
Question: Identify the acoustic scene in the audio.\nOptions:\n(A) bus\n(B) cafe\n(C) car\n(D) crowdedindoor\n(E) elevator\n(F) kitchen\n(G) park\n(H) residentialarea\n(I) restaurant\n(J) restroom\n(K) street\n(L) subway\n(M) subwaystation\n.Answer with the option's letter from the given choices directly and only give the best option.
Model Answer: D
Ground Truth Answer: CrowdedIndoor
Output:
Result: yes

Task:
Input:
Question: {question}
Model Answer: {prediction}
Ground Truth Answer: {answer}
Output:
"""  # noqa


class AudioMQADataset(AudioBaseDataset):
    INTERACTIVE = 'Audio-analysis'
    TASK = 'MQA'
    LANG = None

    def evaluate(self, eval_file, dump_judge=True, method='default'):
        if method == 'vb-mcq':
            metrics, judge_results = self.evaluate_vb_mcq(
                eval_file)
            judge_model_name = 'vb-mcq'
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

    def extract_answer_vb_mcq(self, response):
        response = response.lower()
        if response.startswith('<1>') or response.startswith('<2>') or response.startswith('<3>'):
            response = response[3:].strip()
        for template in [
            '答案是[CHOICE]',
            '答案是 [CHOICE]',
            '答案是选项[CHOICE]',
            '答案应该是[CHOICE]',
            '答案应该是 [CHOICE]',
            '答案就是选项[CHOICE]',
            '答案是‘[CHOICE]',
            '是[CHOICE]：',
            '答案选[CHOICE]',
            '[CHOICE]是正确',
            '选项[CHOICE]是最合适的',
            'answer is: **[CHOICE]',
            'answer is **[CHOICE]',
            'the answer to the question is: **[CHOICE]',
            'the answer to the multiple-choice question is **[CHOICE]',
            "the answer is '[CHOICE]'",
            '[CHOICE] is the best answer',
            'the answer is [CHOICE]',
            'the correct answer is [CHOICE]',
            'would select [CHOICE]',
            'would choose [CHOICE]',
            'would select option [CHOICE]',
            'would choose option [CHOICE]',
            'is \"[CHOICE]\"',
            'is \"[CHOICE].',
            'is: **[CHOICE])',
            'is **[CHOICE],',
            'is **[CHOICE]:',
            'is **[CHOICE])',
            'is: **[CHOICE].',
            'is: **[CHOICE]:',
            'is **[CHOICE].',
            'be **[CHOICE],',
            'is: **[CHOICE]**',
            'is therefore option **[CHOICE]:',
            'is: \n\n**[CHOICE])',
            'as **[CHOICE]:',
            'be **[CHOICE])',
            'be **[CHOICE]:',
            'is: \n\n**[CHOICE]**',
            'suggests **[CHOICE])',
            'be option **[CHOICE]:',
            'with **[CHOICE])',
            "is typically \"[CHOICE])",
            'be to **[CHOICE])',
            'is: \n\n[CHOICE])',
            'is likely to be: **[CHOICE].',
            'is **[CHOICE] (',
            'is option **[CHOICE]**',
            'is likely **[CHOICE]**',
            'is:\n**[CHOICE].',
            'is:\n\n**[CHOICE].',
            'would be [CHOICE]',
            'would be option [CHOICE]',
            'would be ([CHOICE])',
            'would be option ([CHOICE])',
            'is [CHOICE],',
            'is typically [CHOICE],',
            'is typically [CHOICE].',
            "i'd say [CHOICE].",
            'option [CHOICE].',
            'option [CHOICE]:',
            'option [CHOICE],',
            'the answer is:\n**[CHOICE]',
            'is [CHOICE]:',
            'is [CHOICE].',
            'is [CHOICE],',
            'is: [CHOICE].',
            'is ([CHOICE])',
            'is:\n**[CHOICE])',
            'is likely **[CHOICE]:',
            'is the **[CHOICE])',
            ':\n[CHOICE].',
            ':\n[CHOICE])',
            ':\n[CHOICE],',
            ': \n[CHOICE].',
            ':  \n[CHOICE].',
            ':\n\n[CHOICE].',
            ':\n\n[CHOICE])',
            'is most likely **[CHOICE]:',
            ':\n\n[CHOICE],',
            ': \n\n[CHOICE].',
            'is option [CHOICE],',
            '([CHOICE]) would be',
            'is ([CHOICE]).',
            'is [CHOICE])',
            'is: [CHOICE])',
            'is:\n\n[CHOICE]:',
            'is: **[CHOICE],',
            '(option [CHOICE])',
            'answer is ([CHOICE])',
            "select option \"[CHOICE]\"",
            'is: [CHOICE]',
            'is typically **[CHOICE],',
            'is **[CHOICE]**',
            "is likely '[CHOICE]'",
            "is option '[CHOICE]'",
            'is:\n**[CHOICE]:',
            'is \\( \\boxed{[CHOICE] ',
            "would be '[CHOICE]'",
            'is the **[CHOICE]** ',
            'question is [CHOICE] (',
            'is:\n\n**[CHOICE])',
            'closest to option **[CHOICE]**',
            'is most likely **[CHOICE])',
            "the answer to the question is '[CHOICE]'",
            'question is **[CHOICE]**',
            "known as '[CHOICE]'",
            "is '[CHOICE])",
            'is typically **[CHOICE]:',
            'is \\( \\boxed{\\text{[CHOICE]}} \\)',
            'is \\( \\text{[CHOICE]) }',
            'is \\( \\text{[CHOICE]} \\)',
            'is \\( \\text{[CHOICE]:',
            'is \\( \\text{[CHOICE])',
            'is \\(\\text{[CHOICE].',
            'is:\n\n**[CHOICE]',
            'is \\( \\text{[CHOICE].}',
            'is \\( \\text{[CHOICE].',
            'is \\( \\boxed{[CHOICE]}',
            'is:\n\\[ \\boxed{\\text{[CHOICE]}}',
            'is:\n\\[ \\text{[CHOICE])',
            'is:\n\n\\[ \\text{[CHOICE])',
            'is \\( \\textbf{[CHOICE])',
            'is \\( \\text{[CHOICE]}',
            'is: \\( \\text{[CHOICE].',
            'corresponds to:\n- **[CHOICE]:',
            'would be: **[CHOICE]**.',
            'is \\( [CHOICE] \\)',
            'is:\n**[CHOICE] ',
            'corresponds to option **[CHOICE]**',
            'be **[CHOICE]**',
            'be: \n\n[CHOICE])',
            'is:\n\\[ \\boxed{[CHOICE]}',
            'is:  \n**[CHOICE]:',
            'is: \\( \\text{[CHOICE])',
            'is likely: **[CHOICE],',
            'is } \\mathbf{[CHOICE].',
            'is \\( \\boxed{[CHOICE])',
            'is \\( \\textbf{[CHOICE]}',
            'is \\([CHOICE]\\)',
            'is:\n  \n**[CHOICE]:',
            'is option **[CHOICE] ',
            'is:\n\\( \\textbf{[CHOICE].',
            'is \\( \\mathbf{[CHOICE]}',
            'was option **[CHOICE]**',
            "is likely \"[CHOICE])",
            'option **[CHOICE]:',
            "is \"[CHOICE])",
            'is most likely **[CHOICE],',
            'is often **[CHOICE]:',
            'is:  \n[CHOICE])',
            ' [CHOICE].',
            ' [CHOICE],',
            ' [CHOICE]:',
            ' [CHOICE])',
            '**[CHOICE].',
            '**[CHOICE])',
            "\"[CHOICE].",
            "\"[CHOICE],",
            "\"[CHOICE]:",
            '([CHOICE])',
            "\"[CHOICE]\"",

        ]:
            for choice in ['a', 'b', 'c', 'd']:
                if template.replace('[CHOICE]', choice) in response:
                    return choice.upper()
        for choice in ['a', 'b', 'c', 'd']:
            if response == choice:
                return choice.upper()
            for punc in ['.', ',', ':', ')']:
                if response.startswith(choice+punc):
                    return choice.upper()

        if 'would be a.' in response:
            return 'A'
        elif 'would be \"a.' in response:
            return 'A'
        elif 'the best option from the given choices would be a scorpion (a)' in response:
            return 'A'
        else:
            # print({response})
            # print('====')
            return None

    def evaluate_vb_mcq(self, eval_file):
        df = pd.read_json(eval_file, lines=True)
        metrics = {}
        judge_results = {}
        for task, group in df.groupby('subset'):
            if task == 'sentiment':
                continue
            ground_truth = group['answer'].astype(str).to_list()
            preds = group['prediction'].astype(str).to_list()
            preds = [self.extract_answer_vb_mcq(pred) for pred in preds]
            cnt = 0
            for idx in range(len(preds)):
                if preds[idx] is None:
                    preds[idx] = random.choice(['A', 'B', 'C', 'D'])
                    cnt += 1
            results = []
            for idx, (pred, gt) in enumerate(zip(preds, ground_truth)):
                if pred is None:
                    results.append((idx, None))
                elif pred == gt:
                    results.append((idx, 'yes'))
                else:
                    results.append((idx, 'no'))
            task_result, judge_result = self.collect_acc(results, group)
            metrics[task] = task_result
            print(f'{task} result: {task_result}')
            judge_results[task] = judge_result
        return metrics, judge_results

    def evaluate_llm(self, eval_file, judge_model=None):
        df = pd.read_json(eval_file, lines=True)
        metrics = {}
        judge_results = {}
        for task, group in df.groupby('subset'):
            if task == 'sentiment':
                continue
            question = self.get_LLM_query(group)
            gt = group['answer'].astype(str).to_list()
            pred = group['prediction'].astype(str).to_list()

            if self.DATASET_NAME == 'tut2017':
                judge_prompt = TUT2017_SINGLE_CHOICE_PRPMPT
            elif self.DATASET_NAME == 'cochlscene':
                judge_prompt = COCHLSCENE_SINGLE_CHOICE_PRPMPT
            else:
                judge_prompt = SINGLE_CHOICE_PRPMPT

            results = self.run_llm_judge(
                judge_model, judge_prompt, pred=pred, gt=gt, question=question)
            task_result, judge_result = self.collect_acc(results, group)
            metrics[task] = task_result
            print(f'{task} result: {task_result}')
            judge_results[task] = judge_result
        return metrics, judge_results


class MMAUTestMini(AudioMQADataset):
    DATASET_NAME = 'mmau-test-mini'
    DATASET_SERIES = 'MMAU'
    AUDIO_TYPE = 'AudioEvent'


class OpenBookQA(AudioMQADataset):
    DATASET_NAME = 'openbookqa'
    DATASET_SERIES = 'VoiceBench'
    AUDIO_TYPE = 'Speech'
    INTERACTIVE = 'Audio-QA'


class MMSU(AudioMQADataset):
    DATASET_NAME = 'mmsu'
    DATASET_SERIES = 'VoiceBench'
    AUDIO_TYPE = 'Speech'
    INTERACTIVE = 'Audio-QA'


class MELD(AudioMQADataset):
    DATASET_NAME = 'MELD'
    DATASET_SERIES = 'MELD'
    AUDIO_TYPE = 'Speech'


class Nonspeech7k(AudioMQADataset):
    DATASET_NAME = 'Nonspeech7k'
    DATASET_SERIES = 'Nonspeech7k'
    AUDIO_TYPE = 'AudioEvent'


class TUT2017(AudioMQADataset):
    DATASET_NAME = 'TUT2017'
    DATASET_SERIES = 'TUT2017'
    AUDIO_TYPE = 'AudioEvent'


class Vocalsound(AudioMQADataset):
    DATASET_NAME = 'VocalSound'
    DATASET_SERIES = 'VocalSound'
    AUDIO_TYPE = 'AudioEvent'


class Cochlscene(AudioMQADataset):
    DATASET_NAME = 'CochlScene'
    DATASET_SERIES = 'CochlScene'
    AUDIO_TYPE = 'AudioEvent'
