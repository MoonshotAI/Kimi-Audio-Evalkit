# Kimi-Audio-Evalkit

[中文版本](README_zh.md)

## Introduction

Kimi-Audio-Evalkit is an evaluation framework designed for audio large language models. Based on Kimi-Audio-Evalkit, you can quickly implement your own models or datasets and conduct fair comparisons with other open-source models.

Our work [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio-Evalkit) is evaluated using this framework.

See [Leaderboard](./LEADERBOARD.md) for current results.

## Getting Started

### Step1: Get the Code

```bash
git clone https://github.com/MoonshotAI/Kimi-Audio-Evalkit.git
cd Kimi-Audio-Evalkit 
git submodule update --init --recursive
```

### Step2: Prepare Environment

You can directly use our pre-built Docker image. If you need to update the environment, you can modify the Dockerfile and rebuild it.
```bash
docker pull moonshotai/almevalkit:v0.4
```
Typically, you need to mount a local directory as the workspace to ensure evaluation results persist after container exit:
```bash
docker run -it -v $(pwd):/app moonshotai/almevalkit:v0.4 bash
```

### Step3: Get Datasets

Most datasets used by ALMEvalKit can be downloaded using our included tools. Some datasets cannot be fully automated. Please refer to [Download Datasets](./data/README.md) for details.
For datasets on Hugging Face, we will soon provide a more direct usage method. Please stay tuned for updates.

### Step4: Configure config.yaml
You may need to fill in several fields in the config.yaml in the root directory to help us locate your data source. By default, datasets will be downloaded to the data/ directory under the current directory. If you downloaded them elsewhere, please enter the root directory in the dataset_root field.
```yaml
DATASETS:
  dataset_root: "/path/to/your/dataset/root"
```
### Step5: Evaluation

run_audio.sh is the entry point for evaluation. You can get help using `--help`

For example, to evaluate Kimi-Audio on all datasets:
```
bash run_audio.sh --model Kimi-Audio --data all --skip-eval
```
By default, inference results, evaluation results, and metric reports will be generated in the eval_results directory under the current directory. You can change this behavior by passing --work-dir.

Using --skip-eval allows the model to only perform inference without evaluation, which helps keep your GPU running efficiently.
After inference is complete, you can run the command again to start evaluation. You can add the --reeval parameter to force re-evaluation of the dataset, which won't trigger re-inference but will regenerate the metric report.

Note: Our default LLM method is gpt-4o-mini. You need to set your own API KEY to enable it. We will support more evaluation models in the future.
```
export OPENAI_API_KEY=your_api_key
bash run_audio.sh --model Kimi-Audio --data all --reeval
```

Currently supported models, datasets, and evaluation models are listed below:

**Models**

- **Baichuan Series**: Baichuan-Audio-Base, Baichuan-Audio-Instruct
- **Qwen Series**: Qwen2-Audio-7B, Qwen2-Audio-7B-Instruct, Qwen2.5-Omni-7B
- **GLM Series**: GLM4-Voice
- **Others**: StepAudio, Kimi-Audio

**Datasets**

| Dataset Category | Datasets |
|-----------------|----------|
| ASR | LibriSpeech, Fleurs-zh, Fleurs-en, AISHELL-1, AISHELL-2, WenetSpeech |
| MQA | mmau-test-mini, openbookqa, mmsu, MELD, Nonspeech7k, TUT2017, VocalSound, CochlScene |
| OpenQA | alpacaeval_full, commoneval, advbench, ifeval |
| RefQA | ClothoAQA, sd-qa, OpenAudioBench |

- For more information about dataset types, ownership, etc., please check the implementation of the relevant datasets.

## Adding Datasets

We believe the greatest value of ALMEvalKit is not in reproducing existing results, but in providing a simple mechanism to help users add their own datasets and models, and conduct fair comparisons with other model results.

We strongly recommend first reading [Dataset Definition](./almeval/datasets/base.py) to understand how we classify datasets. This will help you correctly set the meta information for new datasets, ensuring they are used appropriately.

To add a dataset, you need to write a few lines of code to prepare a jsonl file named dataset_name.jsonl for ALMEvalKit. Each line of the jsonl is a json record, and we require each line to have the following fields:
```
{
    "index": int, # unique identifier for a piece of data
    "audio_path": str | list[str], # audio location
    "question": str, # question or instruction for the audio, e.g., "Please transcribe the audio content into text". Set to empty if not needed
    "answer": str, # ground truth answer. Set to empty if not needed (e.g., for Open-QA)
    "subset": str, # subset name. Sometimes a dataset can be split into several subsets, which will be evaluated separately and reported independently. If you don't have subsets, use the dataset name
}

For Audio-QA type datasets, we require an additional "audio_content" field to provide the content in text form for LLM evaluation models to assess answer correctness:
{
    "index": int, # unique identifier for a piece of data
    "audio_path": str | list[str], # audio location
    "question": str, # question or instruction for the audio
    "audio_content": str, # text form of the audio
    "answer": str, # ground truth answer
    "subset": str, # subset name
}
```
[Download Dataset](./data/download_benchmark.py) shows how we download & process data, which you can refer to.

After completing this file, you can add your dataset to the appropriate category. Generally, by inheriting the parent class of that category and filling in some fields, your dataset will be ready to use. For example:
```
class Vocalsound(AudioMQADataset):
    DATASET_NAME = 'VocalSound'
    DATASET_SERIES = 'VocalSound'
    AUDIO_TYPE = 'AudioEvent'
```
This indicates that the vocalsound dataset is an MQA dataset (multiple-choice questions), it belongs to the vocalsound dataset series, its AUDIO_TYPE is marked as "AudioEvent", indicating that this dataset is related to sound events (non-speech), which will affect some model evaluation behaviors during evaluation.

If you download it to the dataset cache directory, you can now evaluate this dataset on any model:
```bash
bash run_audio.sh --model Kimi-Audio --data vocalsound
```
If you saved this file elsewhere, please tell us in config.yaml:
```yaml
DATASETS:
  dataset_root: "/path/to/your/dataset/root"
  datasets:
    #example:
    example: "/path/to/your/dataset/example.jsonl"
    Vocalsound: "/path/to/your/dataset/VocalSound.jsonl"
```

## Adding Models

Evaluating your model in ALMEvalKit is also very easy. You only need to implement the generate_inner method, whose signature is:

```
def generate_inner(self, msg:dict) -> (str, str)
```
**msg** is a piece of data from the dataset, with the following format:
```python
{
    "index": int, # the index of a piece of data in the dataset, as above
    "audio": list[str] # in most cases, the length is 1, and audio[0] can be used to get the audio for this piece of data
    "text": str # the "question" field of the data, may be empty
    "meta": dict # dataset meta information, such as audio_type, name, task, etc. If a piece of data has a meta field, it will also be included here
}
```

This function returns prompt:str, result:str, where prompt is the actual text sent to the model for inference, and result is the model's inference result.

**Note** "The actual text sent to the model for inference" is not necessarily equal to `msg['text']`, as we can set rules at runtime to modify it. Usually, we implement a `get_prompt(msg) -> text` to handle this.

**The best way to add a model is to copy an already implemented model and follow its pattern**

## Call for Contribution

We hope the community can work together to build a fair, efficient, and unified audio large language model evaluation framework in the following aspects:

- Add features, fix bugs, improve code quality and usability
- Support more models and datasets
- Improve readability, contribute examples and docs

Due to limited information, we cannot find the best prompts for each model across different tasks/datasets. We also welcome the community to provide best practices, making our leaderboard better reflect the model's true capabilities.

We also recommend that you use pre-commit hooks to auto-format your code. see [pre-commit](https://pre-commit.com/)

