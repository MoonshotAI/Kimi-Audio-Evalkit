
## Download Datasets

### WenetSpeech
1. Fill out the Google Form to request access to the [WenetSpeech](https://huggingface.co/datasets/wenet-e2e/wenetspeech) dataset.

2. Obtain your [Hugging Face API token](https://huggingface.co/settings/tokens) from Hugging Face Account Settings.

3. Run the following command in your terminal to log in:

```bash
cd data
huggingface-cli login
```

4. Run the following command in your terminal to download dataset:

```bash
python download_benchmark.py --output-dir ~/downloaded_datasets --dataset WenetSpeech
```

### Others

1. Run the following command in your terminal to download all dataset(If you only want to download a single dataset, the dataset name is as follows: VoiceBench, OpenAudioBench, LibriSpeech, WenetSpeech, Fleurs, AISHELL-1, AISHELL-2, MMAU, ClothoAQA, VocalSound, Nonspeech7k, MELD, TUT2017, CochlScene)

```bash
python download_benchmark.py --output-dir ~/downloaded_datasets --dataset all
```
