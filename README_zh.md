# Kimi-Audio-Evalkit

[English Version](README.md)

## 介绍

Kimi-Audio-Evalkit是一个为音频大模型评测设计的评测框架，基于Kimi-Audio-Evalkit，你可以快速实现自己的模型或数据集，并公平的与其他开源模型进行比对。

我们的工作[Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio-Evalkit)基于此框架评测。

[Leaderboard](./LEADERBOARD.md)是目前的评测结果。

## 开始评测

### Step1: 获取代码

```bash
git clone https://github.com/MoonshotAI/Kimi-Audio-Evalkit.git
cd Kimi-Audio-Evalkit
git submodule update --init --recursive
```

### Step2: 准备环境

你可以直接使用我们预先build好的镜像，如果你需要更新镜像环境，可以修改Dockerfile后重新构造
```bash
docker pull moonshotai/almevalkit:v0.4
```
通常情况你需要mount本地目录，并将其作为工作目录，以便评测结果在容器退出后仍然存在
```bash
docker run -it -v $(pwd):/app moonshotai/almevalkit:v0.4 bash
```

### Step3: 获取数据集

ALMEvalKit所用的大部分数据集都可以通过我们附带的工具下载，有些数据集不能全自动执行，具体请参阅[下载数据集](./data/README.md)
对位于huggingface的数据集，我们很快将提供更直接的使用方式，请关注更新。

### Step4: 配置config.yaml
你也许需要填写根目录下的config.yaml中的若干字段，帮助我们找到你的数据源。默认情况下，数据集会被下载到当前目录的data/downloaded_datasets下，如果你下载到了其他地方，请将根目录填入dataset_root字段。
```yaml
DATASETS:
  dataset_root: "/path/to/your/dataset/root"
```

### Step5: 评测

run_audio.sh为评测入口，你可以通过`--help`取得帮助

例如，我们希望跑Kimi-Audio在全部数据集上的结果：
```
bash run_audio.sh --model Kimi-Audio --dataset all --skip-eval
```
默认情况下，推理结果文件、评测结果文件、指标报告文件将生成在当前目录的eval_results目录下，你可以通过传递--work-dir改变这一行为。

使用--skip-eval可以让模型只推理，不评测，这样有助于保持你的GPU高效运转。
推理完毕后，你只需要重新运行一次，即可展开评测，你可以通过添加--reeval参数来强制对数据集重新评测，这不会触发重新推理，但会重新生成指标报告。

Note: 我们默认的LLM方式是gpt-4o-mini，你需要设定你自己的API KEY来启用。未来我们将支持更多评测模型。
```
export OPENAI_API_KEY=your_api_key
bash run_audio.sh --model Kimi-Audio --dataset all --reeval
```

目前已经支持的模型、数据集和评测模型列表如下

**模型**

- **Baichuan Series**: Baichuan-Audio-Base, Baichuan-Audio-Instruct
- **Qwen Series**: Qwen2-Audio-7B, Qwen2-Audio-7B-Instruct, Qwen2.5-Omni-7B
- **GLM Series**: GLM4-Voice
- **Others**: StepAudio, Kimi-Audio

**数据集**
| 数据集类型 | 数据集 |
|-----------------|----------|
| ASR | LibriSpeech, Fleurs-zh, Fleurs-en, AISHELL-1, AISHELL-2, WenetSpeech |
| MQA | mmau-test-mini, openbookqa, mmsu, MELD, Nonspeech7k, TUT2017, VocalSound, CochlScene |
| OpenQA | alpacaeval_full, commoneval, advbench, ifeval |
| RefQA | ClothoAQA, sd-qa, OpenAudioBench |

- 数据集的类型、归属等更多信息，可以查看相关数据集的实现。

## 添加数据集

我们相信ALMEvalKit的最大价值不是复现某个已有结果，而是提供一种简单的机制帮助用户添加自己的数据集和模型，并能够与其他模型结果公平比较。

我们强烈建议首先阅读[数据集的定义](./almeval/datasets/base.py)了解我们如何对数据集分类，这将帮助你正确的设定新数据集的meta信息，使它们被更正确的使用。

要添加数据集，你需要写几行代码，为ALMEvalKit准备一个名为dataset_name.jsonl的jsonl文件。jsonl的每一行是一个json记录，我们要求每一行必须具有的字段是：
```
{
    "index": int, # 一条数据的唯一标识
    "audio_path": str | list[str], # 音频位置
    "question": str, # 针对音频的问题或指令，例如"请将音频内容转写为文字"，如果你不需要此字段，请设为空
    "answer": str, # ground truth答案，如果你不需要此字段（如Open-QA），请设为空 
    "subset": str, # 子数据，有时候一个数据集可以被切分为若干个子集，这些子集将被分别评估，独立汇报结果。如果你没有子数据集，填数据集名字即可
}

对于Audio-QA类的数据集，我们要求额外增加一个"audio_content"字段，以文字形式写出内容，以便交给LLM评测模型评测答案是否正确。
{
    "index": int, # 一条数据的唯一标识
    "audio_path": str | list[str], # 音频位置
    "question": str, # 针对音频的问题或指令，例如"请将音频内容转写为文字"，如果你不需要此字段，请设为空
    "audio_content": str, # 音频的文本形式
    "answer": str, # ground truth答案，如果你不需要此字段（如Open-QA），请设为空 
    "subset": str, # 子数据，有时候一个数据集可以被切分为若干个子集，这些子集将被分别评估，独立汇报结果。如果你没有子数据集，填数据集名字即可
}
```
[下载数据集](./data/download_benchmark.py)表明了我们如何下载&处理数据，你可以拿来参考。

完成此文件后，你可以将你的数据集添加到适当的类别下，一般而言，继承此类别的父类并填写一些字段后，你的数据集就可用了。例如：
```
class Vocalsound(AudioMQADataset):
    DATASET_NAME = 'VocalSound'
    DATASET_SERIES = 'VocalSound'
    AUDIO_TYPE = 'AudioEvent'
```
这表明，数据集vocalsound是一个MQA数据集（单选题），它所属的数据集系列是vocalsound，它的AUDIO_TYPE标记为"AudioEvent"，说明此数据集是与声音事件（非语音）有关的数据集，这将会在评测时影响一些模型的评测行为。

如果你将其下载到数据集缓存目录下，现在你就可以在任意模型上评测此数据集了
```bash
bash run_audio.sh --model Kimi-Audio --data vocalsound
```
如果你将此文件保存在了其他位置，请在config.yaml中告诉我们
```yaml
DATASETS:
  dataset_root: "/path/to/your/dataset/root"
  datasets:
    #example:
    example: "/path/to/your/dataset/example.jsonl"
    Vocalsound: "/path/to/your/dataset/VocalSound.jsonl"
```

## 添加模型

在ALMEvalKit评测你的模型也十分容易，你只需要实现generate_inner方法即可，此方法的签名是：

```
def generate_inner(self, msg:dict) -> (str, str)
```
**msg** 就是从数据集中的一条数据，它的格式是：
```python
{
    "index": int, # 即数据集中一条数据的index，见上
    "audio": list[str] # 大部分情况下长度是1，取audio[0]即可获得此条数据的音频
    "text": str # 即数据的"question"字段，可能为空
    "meta": dict # 数据集的meta信息，如audio_type, name，task等存在这里，如果数据集的一条数据有meta字段，也将会被吸入此字段中
}
```

此函数的返回是 prompt:str, result:str，prompt为实际送入模型推理的文本，result为模型推理结果。

**注意** "实际送入模型推理的文本"不一定等于`msg['text']`，因为我们可以在运行时设定规则篡改它，通常我们会实现一个`get_prompt(msg) -> text`来做这件事。

**添加一个模型的最好方式就是copy一个已经实现的模型照猫画虎**

## Call for contribution

我们希望社区在如下方面共建一个公平、高效、统一的音频大模型评测框架

- 增加功能，修改bug，提高代码质量和易用性
- 支持更多模型和数据集
- 提高可读性，贡献examples和docs

受限于我们所掌握的信息，我们无法为每个模型找到不同任务/数据集下的最佳prompt，我们也欢迎社区提供最佳实践，使得我们的leaderboard能够更加真实的反应模型的极限能力。

我们推荐使用[pre-commit](https://pre-commit.com/)来自动格式化你的代码，使你的代码规范与项目保持一致。
