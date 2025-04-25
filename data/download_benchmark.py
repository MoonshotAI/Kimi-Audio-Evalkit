#!/usr/bin/env python3
import os
import json
import argparse
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Split
import soundfile as sf
import pandas as pd
import csv
import subprocess
import jsonlines
import zipfile
from loguru import logger
from tqdm import tqdm
import gdown
import requests


class DatasetDownloader(ABC):
    """Base class for all dataset downloaders."""
    
    def __init__(self, output_dir: str):
        self.metadata = {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    @abstractmethod
    def download(self) -> bool:
        """Download the dataset files.
        Returns:
            bool: True if download was successful, False otherwise.
        """
        pass
    
    def run(self, jsonl_path: Optional[str] = None) -> bool:
        """Download and process the dataset.
        
        Args:
            jsonl_path: Optional custom path for the JSONL output.
                        If None, it will use the default name in the output directory.
          
        Returns:
            bool: True if both download and processing were successful.
        """
        if not jsonl_path:
            jsonl_path = os.path.join(self.output_dir, f"{self.__class__.__name__}.jsonl")
            
        success = self.download()
        if not success:
            print(f"Failed to download {self.__class__.__name__}")
            return False
            
        return True


class VoiceBenchDownloader(DatasetDownloader):
    """Downloader for VoiceBench dataset."""
    
    def download(self) -> bool:
        # TODO: Implement download logic for VoiceBench
        print("Downloading VoiceBench dataset...")
        # Load the VoiceBench dataset
        all_subsets = ['alpacaeval_full', 'commoneval', 'sd-qa', 'ifeval', 'advbench', 'openbookqa', 'mmsu']
        for subset in all_subsets:
            metadata_path = os.path.join(self.output_dir, f"{subset}.jsonl")
            if os.path.exists(metadata_path):
                print(f"Skipping {subset} dataset because it already exists")
                continue
            dataset = load_dataset("hlt-lab/voicebench", subset)
            self.metadata[subset] = []
            subset_dir = os.path.join(self.output_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)
            if subset == 'sd-qa':
                dataset_keys = ['usa']
            else:
                dataset_keys = list(dataset.keys())
            index = 0
            for dataset_key in dataset_keys:
                for item in dataset[dataset_key]:
                    audio_path = os.path.join(subset_dir, f"{index}.wav")
                    sf.write(audio_path, item['audio']['array'], item['audio']['sampling_rate'])
                    self.metadata[subset].append({
                        'index': index,
                        'question': "", # no question since in VoiceBench, the question is in audio
                        'audio_content': item['prompt'],
                        'audio_path': audio_path,
                        'subset': subset,
                        'task_type': "audio2text",
                    })
                    # rename some keys
                    for key in item.keys():
                        if key not in ["audio", "prompt", "key"]:
                            if key == "instruction_id_list":
                                self.metadata[subset][-1]["instruction"] = item[key]
                            elif key == "kwargs":
                                self.metadata[subset][-1]["instruction_kwargs"] = item[key]
                            elif key == "reference":
                                self.metadata[subset][-1]["answer"] = item[key]
                            else:
                                self.metadata[subset][-1][key] = item[key]
                    index += 1
            print(f"Downloaded {subset} dataset")
            # dump metadata to jsonl
            
            with open(metadata_path, 'w', encoding="utf-8") as f:
                for metadata in self.metadata[subset]:
                    f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
            print(f"Completed processing {subset} dataset. Metadata saved to {metadata_path}")
        return True


class OpenAudioBenchDownloader(DatasetDownloader):
    """Downloader for OpenAudioBench dataset."""
    
    def download(self) -> bool:
        print("Downloading OpenAudioBench dataset...")

        if not os.path.exists(os.path.join(self.output_dir, "OpenAudioBench")):
            result = os.system(f"git clone https://huggingface.co/datasets/baichuan-inc/OpenAudioBench {self.output_dir}/OpenAudioBench")
            if result != 0:
                raise RuntimeError("Failed to clone OpenAudioBench repository, is git lfs installed?")
        else:
            print("OpenAudioBench repository already exists")

        all_subsets = ["llama_questions", "web_questions", "reasoning_qa", "alpaca_eval", "trivia_qa"]

        self.metadata["openaudiobench"] = []
        index = 0
        metadata_path = os.path.join(self.output_dir, f"OpenAudioBench.jsonl")
        if os.path.exists(metadata_path):
            print(f"Skipping OpenAudioBench dataset because it already exists")
            return True

        for subset in all_subsets:
            if subset == "alpaca_eval":
                audio_filename_key = "audio_filename"
                question_key = "instruction"
                answer_key = "output"
            elif subset == "trivia_qa":
                audio_filename_key = "audio_filename"
                question_key = "question"
                answer_key = "answer_normalized_aliases"
            elif subset == "reasoning_qa":
                audio_filename_key = "audio_filename"
                question_key = "Prompt"
                answer_key = "参考答案"
            elif subset == "web_questions":
                audio_filename_key = "audio_filename"
                question_key = "question"
                answer_key = "answers"
            elif subset == "llama_questions":
                audio_filename_key = "audio_filename"
                question_key = "Questions"
                answer_key = "Answer"
            else:
                raise ValueError(f"Unknown subset: {subset}")
            meta_csv_path = os.path.join(self.output_dir, "OpenAudioBench/eval_datas", subset, f"{subset}.csv")
            df = pd.read_csv(meta_csv_path)
            for _, row in df.iterrows():
                audio_path = os.path.join(self.output_dir, "OpenAudioBench/eval_datas", subset, "audios", row[audio_filename_key])
                assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"
                self.metadata["openaudiobench"].append({
                    "index": index,
                    "audio_path": audio_path,
                    "subset": subset,
                    "question": "",
                    "audio_content": row[question_key],
                    "answer": row[answer_key],
                    "task_type": "audio2text",
                })
                index += 1
            
        with open(metadata_path, 'w', encoding="utf-8") as f:
            for metadata in self.metadata["openaudiobench"]:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        print(f"Completed processing OpenAudioBench dataset. Metadata saved to {metadata_path}")
        return True


class LibrispeechDownloader(DatasetDownloader):
    """Downloader for Librispeech dataset."""
    
    def download(self) -> bool:
        librispeech_dir = os.path.join(self.output_dir, "librispeech")
        if not os.path.exists(librispeech_dir):
            os.makedirs(librispeech_dir, exist_ok=True)
            try:
                original_dir = os.getcwd()
                os.chdir(librispeech_dir)

                # Download test-clean dataset
                download_success = os.system("wget https://us.openslr.org/resources/12/test-clean.tar.gz -O test-clean.tar.gz")
                if download_success != 0:
                    raise RuntimeError("Failed to download test-clean dataset")
                # Download test-other dataset
                download_success = os.system("wget https://us.openslr.org/resources/12/test-other.tar.gz -O test-other.tar.gz")
                if download_success != 0:
                    raise RuntimeError("Failed to download test-other dataset")
                
                # Extract the tar.gz dataset
                extract_success = os.system("tar -xzf test-clean.tar.gz")
                if extract_success != 0:
                    raise RuntimeError("Failed to extract test-clean dataset")
                extract_success = os.system("tar -xzf test-other.tar.gz")
                if extract_success != 0:
                    raise RuntimeError("Failed to extract test-other dataset")
                    
                # Restore original directory
                os.chdir(original_dir)
                
            except Exception as e:
                print(f"Error downloading librispeech dataset: {str(e)}")
                return False
        else:
            print("librispeech dataset already downloaded")
        
        self.metadata["librispeech"] = []
        index = 0
        metadata_path = os.path.join(self.output_dir, f"LibriSpeech.jsonl")
        if os.path.exists(metadata_path):
            print(f"Skipping librispeech dataset because it already exists")
            return True
        
        question = "Please transcribe the spoken content into written text."

        subsets = ["test-clean", "test-other"]

        index = 0
        for subset in subsets:
            subset_dir = os.path.join(self.output_dir, "librispeech/LibriSpeech", subset)
            for spk_folder in tqdm(os.listdir(subset_dir)):
                for chapter_folder in os.listdir(os.path.join(subset_dir, spk_folder)):
                    # get all the flac files in the chapter_folder
                    flac_files = [f for f in os.listdir(os.path.join(subset_dir, spk_folder, chapter_folder)) if f.endswith(".flac")]
                    transcript_path = os.path.join(subset_dir, spk_folder, chapter_folder, f"{spk_folder}-{chapter_folder}.trans.txt")
                    transcript_dict = {}
                    with open(transcript_path, 'r', encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split(" ", 1)
                            assert len(parts) == 2, f"Invalid line: {line}"
                            flac_file = parts[0]
                            transcript = parts[1]
                            transcript_dict[flac_file] = transcript
                    for flac_file in flac_files:
                        audio_path = os.path.join(subset_dir, spk_folder, chapter_folder, flac_file)
                        transcript = transcript_dict[flac_file.split(".")[0]]
                        assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"
                        self.metadata["librispeech"].append({
                            "index": index,
                            "question": question,
                            "audio_path": audio_path,
                            "answer": transcript,
                            "subset": subset.replace("-", "_"),
                            "task_type": "understanding",
                        })
                        index += 1
        
        with open(metadata_path, 'w', encoding="utf-8") as f:
            for metadata in self.metadata["librispeech"]:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        print(f"Completed processing LibriSpeech dataset. Metadata saved to {metadata_path}")
        return True


class WenetspeechDownloader(DatasetDownloader):
    """Downloader for Wenetspeech dataset."""
    
    def download(self) -> bool:
        # TODO: Implement download logic for Wenetspeech
        print("Downloading Wenetspeech dataset...")
        wenetspeech_dir = os.path.join(self.output_dir, "wenetspeech")
        cache_dir = os.path.join(wenetspeech_dir, "cache")
        audio_dir = os.path.join(wenetspeech_dir, "audios")
        metadata_path = os.path.join(self.output_dir, f"WenetSpeech.jsonl")
        if os.path.exists(metadata_path):
            print(f"Skipping wenetspeech dataset because it already exists")
            return True
        
        ws_test_net = load_dataset("wenet-e2e/wenetspeech", "TEST_NET", split="test", trust_remote_code=True, cache_dir=cache_dir)
        ws_test_meeting = load_dataset("wenet-e2e/wenetspeech", "TEST_MEETING", split="test", trust_remote_code=True,  cache_dir=cache_dir)
        
        index = 0
        all_data = []
        skipped = 0
        for ds in [ws_test_net, ws_test_meeting]:
            for item in tqdm(ds):
                bmk_name = 'test_net' if ds == ws_test_net else 'test_meeting'
                audio = item['audio']
                audio_path = audio['path']
                org_audio_path = item['original_full_path'] # data/cuts_TEST_NET.00000000/TES/TEST_NET_Y0000000009_26OUqTzrzv0_S00624.wav
                audio_save_dir = os.path.join(audio_dir, os.path.dirname(org_audio_path))
                os.makedirs(audio_save_dir, exist_ok=True)
                new_audio_path = os.path.join(audio_save_dir, os.path.basename(audio_path))
                if not os.path.exists(new_audio_path) and os.path.exists(audio_path):
                    os.system('cp {} {}'.format(audio_path, new_audio_path))

                # 如果没有，跳过
                if not os.path.exists(new_audio_path):
                    skipped += 1
                    logger.warning(f'audio {os.path.basename(audio_path)} not exists, skip. total skipped: {skipped}')
                    continue

                write_item = {
                    'index': index,
                    'audio_path': new_audio_path,
                    'question': '请把这段语音转录成文本。',
                    'answer': item['text'],
                    'subset': bmk_name,
                    "task_type": "understanding"
                }
                all_data.append(write_item)
                index += 1

        with jsonlines.open(metadata_path, 'w') as f:
            for item in all_data:
                f.write(item)
        print(f"Completed processing Wenetspeech dataset. Metadata saved to {metadata_path}")
        return True



class FleursDownloader(DatasetDownloader):
    """Downloader for Fleurs dataset."""
    
    def download(self) -> bool:
        print("Downloading Fleurs dataset...")
        
        # Create directories
        fleurs_dir = os.path.join(self.output_dir, "fleurs")
        os.makedirs(fleurs_dir, exist_ok=True)
        
        # Define language configurations
        lang_configs = [
            {"code": "cmn_hans_cn", "output_file": "Fleurs-zh.jsonl", "question": "请将这段语音转写成文字。"},
            {"code": "en_us", "output_file": "Fleurs-en.jsonl", "question": "Please transcribe this audio."}
        ]
        
        for config in lang_configs:
            lang_code = config["code"]
            output_file = config["output_file"]
            question = config["question"]
            metadata_path = os.path.join(self.output_dir, output_file)
            
            # Skip if already processed
            if os.path.exists(metadata_path):
                print(f"Skipping {lang_code} dataset because it already exists")
                continue
                
            print(f"Processing {lang_code} dataset...")
            
            # Load the dataset
            dataset = load_dataset("google/fleurs", lang_code, trust_remote_code=True)
            
            # Create language directory
            lang_dir = os.path.join(fleurs_dir, lang_code)
            os.makedirs(lang_dir, exist_ok=True)

            # Process the dataset
            self.metadata["fleurs_" + lang_code] = []
            
            # Process test split
            for index, item in tqdm(enumerate(dataset["test"]), desc=f"Processing {lang_code} dataset", total=len(dataset["test"])):
                # Save audio file
                audio_path = os.path.join(lang_dir, f"{index}.wav")
                sf.write(audio_path, item["audio"]["array"], item["audio"]["sampling_rate"])
                
                # Create metadata entry
                self.metadata["fleurs_" + lang_code].append({
                    "index": index,
                    "question": question,
                    "audio_path": audio_path,
                    "answer": item["transcription"],
                    "raw_transcription": item["raw_transcription"],
                    "language": item["language"],
                    "subset": f"fleurs_{lang_code}",
                    "task_type": "understanding",
                })
            
            # Save metadata to jsonl file
            with open(metadata_path, 'w', encoding="utf-8") as f:
                for metadata in self.metadata["fleurs_" + lang_code]:
                    f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
            print(f"Completed processing {lang_code} dataset. Metadata saved to {metadata_path}")
        return True



class Aishell1Downloader(DatasetDownloader):
    """Downloader for Aishell1 dataset."""
    
    def download(self) -> bool:
        aishell1_dir = os.path.join(self.output_dir, "aishell1")
        test_speakers = [
            'S0764', 'S0765', 'S0766', 'S0767', 'S0768', 'S0769', 'S0770',
            'S0901', 'S0902', 'S0903', 'S0904', 'S0905', 'S0906', 'S0907',
            'S0908', 'S0912', 'S0913', 'S0914', 'S0915', 'S0916'
        ]

        if not os.path.exists(aishell1_dir):
            os.makedirs(aishell1_dir, exist_ok=True)
            try:
                original_dir = os.getcwd()
                os.chdir(aishell1_dir)

                # aishell dataset
                download_success = os.system("wget https://us.openslr.org/resources/33/data_aishell.tgz -O data_aishell.tgz")
                if download_success != 0:
                    raise RuntimeError("Failed to download aishell dataset")
                
                # Extract the tar.gz dataset
                extract_success = os.system("tar -xvzf data_aishell.tgz")
                if extract_success != 0:
                    raise RuntimeError("Failed to extract aishell dataset")

                # only extract the test set speaker
                for speaker in test_speakers:
                    extract_success = os.system(f"tar -xvzf data_aishell/wav/{speaker}.tar.gz -C data_aishell/wav")
                    if extract_success != 0:
                        raise RuntimeError(f"Failed to extract aishell dataset {speaker}")

                # Restore original directory
                os.chdir(original_dir)
                
            except Exception as e:
                print(f"Error downloading aishell dataset: {str(e)}")
                return False
        else:
            print("aishell dataset already downloaded")
        
        self.metadata["aishell1"] = []

        metadata_path = os.path.join(self.output_dir, f"AISHELL-1.jsonl")
        if os.path.exists(metadata_path):
            print(f"Skipping aishell1 dataset because it already exists")
            return True
        
        question = "请把这段语音转录成文本。"

        transcript_path = os.path.join(self.output_dir, "aishell1/data_aishell/transcript/aishell_transcript_v0.8.txt")
        # load transcript_dict
        transcript_dict = {}
        with open(transcript_path, 'r', encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                assert len(parts) == 2, f"Invalid line: {line}"
                flac_file = parts[0]
                transcript = parts[1]
                transcript_dict[flac_file] = transcript

        index = 0
        for speaker in test_speakers:
            subset_dir = os.path.join(self.output_dir, "aishell1/data_aishell/wav/test", speaker)
            for wav_file in os.listdir(subset_dir):
                answer = transcript_dict[wav_file.split(".")[0]]
                audio_path = os.path.join(subset_dir, wav_file)
                assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"
                self.metadata["aishell1"].append({
                    "index": index,
                    "audio_path": audio_path,
                    "question": question,
                    "answer": "".join(answer.strip().split()),
                    "subset": "aishell1_test",
                    "task_type": "understanding",
                })
                index += 1
        
        with open(metadata_path, 'w', encoding="utf-8") as f:
            for metadata in self.metadata["aishell1"]:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        print(f"Completed processing Aishell1 dataset. Metadata saved to {metadata_path}")
        return True


class Aishell2Downloader(DatasetDownloader):
    """Downloader for Aishell2 dataset."""
    
    def download(self) -> bool:
        # TODO: Implement download logic for Aishell2
        print("Downloading Aishell2 dataset...")

        aishell2_dir = os.path.join(self.output_dir, "aishell2")
        ios_dir = os.path.join(aishell2_dir, "AISHELL-DEV-TEST-SET/iOS")
        metadata_path = os.path.join(self.output_dir, f"AISHELL-2.jsonl")
        if os.path.exists(metadata_path):
            print(f"Skipping aishell2 dataset because it already exists")
            return True

        if not os.path.exists(aishell2_dir):
            os.makedirs(aishell2_dir, exist_ok=True)
            try:
                original_dir = os.getcwd()
                os.chdir(aishell2_dir)

                download_success = os.system('wget "https://aishell-eval.oss-cn-beijing.aliyuncs.com/TEST&DEV%20DATA.zip" -O test_dev.zip')
                if download_success != 0:
                    raise RuntimeError("Failed to download aishell2 dataset")
                
                # Extract the zip dataset
                extract_success = os.system("unzip test_dev.zip")
                if extract_success != 0:
                    raise RuntimeError("Failed to extract aishell2 dataset")

                os.chdir(ios_dir)
                extract_success = os.system("tar -xvzf  test.tar.gz")
                if extract_success != 0:
                    raise RuntimeError("Failed to extract aishell2 testset")
                    
                # Restore original directory
                os.chdir(original_dir)
                
            except Exception as e:
                print(f"Error downloading aishell2 dataset: {str(e)}")
                return False
        else:
            print("aishell2 dataset already downloaded")

        filename_map = {}
        with open(os.path.join(ios_dir, 'test/wav.scp'), 'r') as f:
            for line in f:
                filename, path = line.strip().split('\t')
                filename_map[filename] = path
        
        bmk_name = 'aishell2_test'
        index = 0
        all_data = []
        with open(os.path.join(ios_dir, 'test/trans.txt'), 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.strip()
                if line:
                    filename, trans = line.split('\t', 1)
                    filename_real = filename_map[filename]
                    audio_path = f'{ios_dir}/test/{filename_real}'
                    answer = trans
                    assert os.path.exists(audio_path)
                
                    write_item = {
                        'index': index,
                        'audio_path': audio_path,
                        'question': '请把这段语音转录成文本。',
                        'answer': answer,
                        'subset': bmk_name,
                        "task_type": "understanding"
                    }
                all_data.append(write_item)
                index += 1

        with jsonlines.open(metadata_path, 'w') as f:
            for item in all_data:
                f.write(item)
        print(f"Completed processing Aishell2 dataset. Metadata saved to {metadata_path}")
        return True
    


class MMAUTestMiniDownloader(DatasetDownloader):
    """Downloader for MMAU-test-mini dataset."""
    
    def download(self) -> bool:
        print("Downloading MMAU-test-mini dataset...")
        
        # Create directories
        mmau_dir = os.path.join(self.output_dir, "MMAU")
        os.makedirs(mmau_dir, exist_ok=True)
        
        # Define the output jsonl path
        metadata_path = os.path.join(self.output_dir, "mmau-test-mini.jsonl")
        
        # Skip if already processed
        if os.path.exists(metadata_path):
            print("Skipping MMAU-test-mini dataset because it already exists")
            return True


        cache_dir = os.path.join(mmau_dir, "data/")
        os.makedirs(cache_dir, exist_ok=True)


        url = "https://drive.google.com/file/d/1fERNIyTa0HWry6iIG1X-1ACPlUlhlRWA/view"
        output_path = os.path.join(cache_dir, "mmau-test-mini-audios.tar.gz")
        gdown.download(url, output_path, quiet=False, fuzzy=True)

        # decompress the tar.gz file
        extract_success = os.system(f"tar -xvzf {cache_dir}/mmau-test-mini-audios.tar.gz -C {cache_dir}/")
        if extract_success != 0:
            raise RuntimeError("Failed to extract mmau-test-mini-audios")

        json_path = os.path.join(cache_dir, "mmau-test-mini.json")
        audio_dir = os.path.join(cache_dir, "test-mini-audios")
        
        # Download JSON file if it doesn't exist
        if not os.path.exists(json_path):
            download_url = "https://raw.githubusercontent.com/Sakshi113/MMAU/main/mmau-test-mini.json"
            
            response = requests.get(download_url)
            if response.status_code == 200:
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"Downloaded MMAU test mini JSON from GitHub")
            else:
                print(f"Failed to download: HTTP {response.status_code}")
                return False
        
        # Process the JSON file and create metadata
        with open(json_path, "r", encoding="utf-8") as f:
            mmau_data = json.load(f)
        
        print(f"Loaded MMAU test mini JSON with {len(mmau_data)} entries")
        
        # Process the dataset
        self.metadata["mmau-test-mini"] = []
        
        # Process each item in the JSON
        for index, item in tqdm(enumerate(mmau_data), desc="Processing MMAU test mini dataset", total=len(mmau_data)):
            # Get audio file path
            audio_id = item["id"]
            audio_file = item.get("audio_file", f"{audio_id}.wav")
            src_audio_path = os.path.join(audio_dir, audio_file)
            
            # Check if audio file exists
            if not os.path.exists(src_audio_path):
                print(f"Warning: Audio file {src_audio_path} does not exist, skipping")
                continue

            question = item["question"] + "\n"

            # choices to (A) X (B) Y (C) Z
            choices = [f"({chr(65 + i)}) {choice}" for i, choice in enumerate(item["choices"])]
            question += " ".join(choices)
            
            # Create metadata entry
            metadata_entry = {
                "index": index,
                "question": question,
                "audio_path": src_audio_path,  # Use the original path
                "choices": item["choices"],
                "answer": item["answer"],
                "dataset": "mmau",
                "subset": item["task"],
                "task_type": "understanding",

                "difficulty": item["difficulty"],
                "category": item["category"],
                "sub-category": item["sub-category"],
            }
            
            self.metadata["mmau-test-mini"].append(metadata_entry)
        
        # Save metadata to jsonl file
        with open(metadata_path, 'w', encoding="utf-8") as f:
            for metadata in self.metadata["mmau-test-mini"]:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
                
        print(f"Completed processing MMAU-test-mini dataset. Metadata saved to {metadata_path}")
        return True
        


class ClothoAQADownloader(DatasetDownloader):
    """Downloader for ClothoAQA dataset."""
    
    def download(self) -> bool:
        print("Downloading ClothoAQA dataset...")
        
        # Create directories
        clotho_dir = os.path.join(self.output_dir, "ClothoAQA")
        os.makedirs(clotho_dir, exist_ok=True)
        
        # Define the output jsonl path
        metadata_path = os.path.join(self.output_dir, "ClothoAQA.jsonl")
        
        # Skip if already processed
        if os.path.exists(metadata_path):
            print("Skipping ClothoAQA dataset because it already exists")
            return True


        cache_dir = os.path.join(clotho_dir, "data/")
        os.makedirs(cache_dir, exist_ok=True)

        url = "https://zenodo.org/records/6473207/files/audio_files.zip?download=1"
        output_path = os.path.join(cache_dir, "audio_files.zip")

        # check if the file exists
        if not os.path.exists(output_path):
            download_success = os.system(f"wget {url} -O {output_path}")
            if download_success != 0:
                raise RuntimeError("Failed to download audio_files")

        # unzip the audio_files.zip
        extract_success = os.system(f"unzip {cache_dir}/audio_files.zip -d {cache_dir}/")
        if extract_success != 0:
            raise RuntimeError("Failed to unzip audio_files")


        os.system(f"git clone https://github.com/GeWu-Lab/MWAFM.git {cache_dir}/MWAFM")
        val_csv_path = os.path.join(cache_dir, "MWAFM/metadata/clotho_aqa_val_clean.csv")
        test_csv_path = os.path.join(cache_dir, "MWAFM/metadata/clotho_aqa_test_clean.csv")

        assert os.path.exists(val_csv_path), f"clotho_aqa_val_clean.csv does not exist"
        assert os.path.exists(test_csv_path), f"clotho_aqa_test_clean.csv does not exist"

        # read the csv files
        val_df = pd.read_csv(val_csv_path)
        test_df = pd.read_csv(test_csv_path)

        self.metadata["clotho_aqa"] = []

        inst_idx = 0

        # process the csv files
        for index, row in val_df.iterrows():
            audio_path = os.path.join(cache_dir, "audio_files", row["file_name"])
            assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"

            question = row["QuestionText"]
            answer = row["answer"]

            self.metadata["clotho_aqa"].append({
                "index": inst_idx,
                "question": question,
                "audio_path": audio_path,
                "answer": answer,
                "subset": "val",
                "task_type": "understanding"
            })
            inst_idx += 1

        for index, row in test_df.iterrows():
            audio_path = os.path.join(cache_dir, "audio_files", row["file_name"])
            assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"

            question = row["QuestionText"]
            answer = row["answer"]

            self.metadata["clotho_aqa"].append({
                "index": inst_idx,
                "question": question,
                "audio_path": audio_path,
                "answer": answer,
                "subset": "test",
                "task_type": "understanding"
            })
            inst_idx += 1

        # save the metadata to jsonl file
        with open(metadata_path, 'w', encoding="utf-8") as f:
            for metadata in self.metadata["clotho_aqa"]:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')

        print(f"Completed processing ClothoAQA dataset. Metadata saved to {metadata_path}")
        return True
    


class VocalsoundDownloader(DatasetDownloader):
    """Downloader for Vocalsound dataset."""
    
    def download(self) -> bool:
        # TODO: Implement download logic for Vocalsound
        print("Downloading Vocalsound dataset...")
        vocalsound_dir = os.path.join(self.output_dir, "vocalsound")
        if not os.path.exists(vocalsound_dir):
            os.makedirs(vocalsound_dir, exist_ok=True)
            try:
                # git clone https://github.com/YuanGongND/vocalsound.git
                result = os.system(f"git clone https://github.com/YuanGongND/vocalsound.git {vocalsound_dir}")
                if result != 0:
                    raise RuntimeError("Failed to clone vocalsound repository")
                # Change to data directory
                original_dir = os.getcwd()
                os.chdir(f"{vocalsound_dir}/data")
                
                # Download vocalsound dataset
                download_success = os.system("wget https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=0 -O vs_release_16k.zip")
                if download_success != 0:
                    raise RuntimeError("Failed to download vocalsound dataset")
                    
                # Unzip the dataset
                unzip_success = os.system("unzip vs_release_16k.zip")
                if unzip_success != 0:
                    raise RuntimeError("Failed to unzip vocalsound dataset")
                    
                # Restore original directory
                os.chdir(original_dir)
                
            except Exception as e:
                print(f"Error downloading vocalsound dataset: {str(e)}")
                return False
        else:
            print("Vocalsound dataset already downloaded")

        meta_json_path = os.path.join(self.output_dir, "vocalsound/data/datafiles/te.json")
        # read meta_json_path
        with open(meta_json_path, 'r', encoding="utf-8") as f:
            meta_json = json.load(f)
        
        self.metadata["vocalsound"] = []
        index = 0
        metadata_path = os.path.join(self.output_dir, f"VocalSound.jsonl")
        if os.path.exists(metadata_path):
            print(f"Skipping vocalsound dataset because it already exists")
            return True
        
        question = "Identify the human vocal sound in the audio.\nOptions:\n(A) Laughter\n(B) Sigh\n(C) Cough\n(D) Throat clearing\n(E) Sneeze\n(F) Sniff\n.Answer with the option's letter from the given choices directly and only give the best option."

        label_to_answer = {
            "/m/01j3sz": "Laughter",
            "/m/07plz5l": "Sigh",
            "/m/01b_21": "Cough",
            "/m/0dl9sf8": "Throat clearing",
            "/m/01hsr_": "Sneeze",
            "/m/07ppn3j": "Sniff"
        }
        
        for index, item in enumerate(meta_json["data"]):
            audio_path = os.path.join(self.output_dir, "vocalsound/data/audio_16k", item["wav"].split("/")[-1])
            assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"
            self.metadata["vocalsound"].append({
                "index": index,
                "question": question,
                "audio_path": audio_path,
                "answer": label_to_answer[item["labels"]],
                "subset": "voice_classification",
                "task_type": "understanding"
            })

        with open(metadata_path, 'w', encoding="utf-8") as f:
            for metadata in self.metadata["vocalsound"]:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        print(f"Completed processing Vocalsound dataset. Metadata saved to {metadata_path}")
        return True
    


class Nonspeech7kDownloader(DatasetDownloader):
    """Downloader for Nonspeech7k dataset."""

    def download(self) -> bool:
        # Ensure the directory exists
        nonspeech7k_dir = self.output_dir
        if not os.path.exists(nonspeech7k_dir):
            os.makedirs(nonspeech7k_dir, exist_ok=True)
        
        try:
            # Download the dataset
            dataset_url = "https://zenodo.org/records/6967442/files/test.zip"
            zip_file_path = os.path.join(nonspeech7k_dir, "test.zip")
            self._download_file(dataset_url, zip_file_path)
            
            # Unzip the dataset
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(nonspeech7k_dir)
            
            # Download the metadata CSV file
            metadata_url = "https://zenodo.org/records/6967442/files/metadata%20of%20test%20set.csv"
            metadata_file_path = os.path.join(nonspeech7k_dir, "metadata_test.csv")
            self._download_file(metadata_url, metadata_file_path)

            # Process the dataset
            return self.process_dataset(nonspeech7k_dir)

        except Exception as e:
            print(f"Error downloading Nonspeech7k dataset: {str(e)}")
            return False

    def _download_file(self, url: str, file_path: str) -> None:
        """Helper method to download a file from a URL."""
        print(f"Downloading {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            raise RuntimeError(f"Failed to download file from {url}")

    def process_dataset(self, dataset_dir: str) -> bool:
        """Process the dataset and create JSONL file."""
        metadata_file = os.path.join(dataset_dir, "metadata_test.csv")
        metadata_path = os.path.join(self.output_dir, "Nonspeech7k.jsonl")

        # Read metadata CSV file
        with open(metadata_file, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            metadata = []

            # Process each row in the CSV
            for index, row in enumerate(csv_reader):
                audio_path = os.path.join(dataset_dir, "test", row["Filename"])
                assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"

                metadata.append({
                    "index": index,
                    "question": "Identify the human vocal sound in the audio.\nOptions:\n(A) Breath\n(B) Cough\n(C) Cry\n(D) Laugh\n(E) Scream\n(F) Sneeze\n(G) Yawn\n.Answer with the option's letter from the given choices directly and only give the best option.",
                    "audio_path": audio_path,
                    "answer": row["Classname"],
                    "subset": "https://freesound.org/",
                    "task_type": "understanding"
                })
        
        # Write the metadata to JSONL file
        with open(metadata_path, 'w', encoding="utf-8") as f:
            for item in metadata:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Completed processing Nonspeech7k dataset. Metadata saved to {metadata_path}")
        return True

    

class MeldDownloader(DatasetDownloader):
    """Downloader for MELD dataset."""

    def dump_jsonl(self, input_csv, video_dir, audio_dir):
        print("extract mp3 from video, it may take a while...")
        os.makedirs(audio_dir, exist_ok=True)

        self.metadata["meld"] = []
        idx = 0
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader):
                dialogue_id = row['Dialogue_ID']
                utterance_id = row['Utterance_ID']
                emotion = row['Emotion']
                sentiment = row['Sentiment']
                mp3_idx = int(row['Sr No.'])

                start_time_str = row['StartTime'] # 00:14:38,127
                end_time_str = row['EndTime'] # 00:14:40,127

                # Skip this sample and warn if duration exceeds 30s
                def str_min_sec_to_sec(str_time):
                    h, m, s = str_time.split(':')
                    h = int(h)
                    m = int(m)
                    s, ms = map(float, s.split(','))
                    return h * 3600 + m * 60 + s + ms / 1000
                
                duration = str_min_sec_to_sec(end_time_str) - str_min_sec_to_sec(start_time_str)
                out_mp3 = os.path.join(audio_dir, f'{mp3_idx}.mp3')
                if duration > 30:
                    logger.warning(f"Dialogue {dialogue_id} utterance {utterance_id} duration {duration} > 30s, skipped")
                    if os.path.exists(out_mp3):
                        os.remove(out_mp3)
                    continue
                
                mp4_path = os.path.join(video_dir, f'dia{dialogue_id}_utt{utterance_id}.mp4')
                
                # Emotion
                
                if not os.path.exists(out_mp3):
                    subprocess.run([
                        'ffmpeg', '-i', mp4_path,
                        '-q:a', '0', '-map', 'a', out_mp3
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                question_emo = ("Identify the predominant emotion in this speech.\nOptions:\n(A) neutral\n(B) joy\n(C) sadness\n(D) anger\n(E) surprise\n(F) fear\n(G) disgust\n.Answer with the option's letter from the given choices directly and only give the best option.")
                self.metadata["meld"].append({
                    'index': str(idx),
                    'audio_path': out_mp3,
                    'question': question_emo,
                    'answer': emotion,
                    'subset': 'emotion',
                    'utterance': row['Utterance'],
                    "task_type": "understanding"
                })
                idx += 1


        return True

    
    def download(self) -> bool:
        print("Downloading MELD dataset...")
        meld_dir = os.path.join(self.output_dir, "meld")
        if not os.path.exists(meld_dir):
            os.makedirs(meld_dir, exist_ok=True)
            try:
                # https://github.com/declare-lab/MELD.git
                result = os.system(f"git clone https://github.com/declare-lab/MELD.git {meld_dir}")
                if result != 0:
                    raise RuntimeError("Failed to clone MELD repository")
                # Change to data directory
                original_dir = os.getcwd()
                os.chdir(meld_dir)
                
                # Download MELD dataset
                download_success = os.system("wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz -O MELD.Raw.tar.gz")
                if download_success != 0:
                    raise RuntimeError("Failed to download MELD dataset")
                    
                # Extract the tar.gz dataset
                extract_success = os.system("tar -xzf MELD.Raw.tar.gz")
                if extract_success != 0:
                    raise RuntimeError("Failed to extract MELD dataset")
                
                # Extract the test.tar.gz
                extract_success = os.system("tar -xzf MELD.Raw/test.tar.gz -C MELD.Raw/")
                if extract_success != 0:
                    raise RuntimeError("Failed to extract MELD testset dataset")
                    
                # Restore original directory
                os.chdir(original_dir)
                
            except Exception as e:
                print(f"Error downloading MELD dataset: {str(e)}")
                return False
        else:
            print("MELD dataset already downloaded")

        meta_json_path = os.path.join(self.output_dir, "meld/MELD.Raw/test_sent_emo.csv")
        video_dir = os.path.join(self.output_dir, "meld/MELD.Raw/output_repeated_splits_test")
        audio_dir = os.path.join(self.output_dir, "meld/audios")

        self.dump_jsonl(meta_json_path, video_dir, audio_dir)

        metadata_path = os.path.join(self.output_dir, f"MELD.jsonl")
        with open(metadata_path, 'w', encoding="utf-8") as f:
            for metadata in self.metadata["meld"]:
                f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        
        print(f"Completed processing MELD dataset. Metadata saved to {metadata_path}")
        return True


class TUT2017Downloader(DatasetDownloader):
    """Downloader for TUT2017 dataset."""

    def download(self) -> bool:
        # Ensure the directory exists
        tut2017_dir = os.path.join(self.output_dir, "TUT2017")
        if not os.path.exists(tut2017_dir):
            os.makedirs(tut2017_dir, exist_ok=True)

        try:
            # Define URLs for audio files and metadata file
            audio_urls = [
                "https://zenodo.org/records/1040168/files/TUT-acoustic-scenes-2017-evaluation.audio.1.zip",
                "https://zenodo.org/records/1040168/files/TUT-acoustic-scenes-2017-evaluation.audio.2.zip",
                "https://zenodo.org/records/1040168/files/TUT-acoustic-scenes-2017-evaluation.audio.3.zip",
                "https://zenodo.org/records/1040168/files/TUT-acoustic-scenes-2017-evaluation.audio.4.zip"
            ]
            
            # Download and unzip audio files if not already done
            for idx, url in enumerate(audio_urls, 1):
                zip_file_path = os.path.join(tut2017_dir, f"audio_{idx}.zip")
                if not os.path.exists(zip_file_path):  # Check if the ZIP file already exists
                    self._download_file(url, zip_file_path)
                if not os.path.exists(os.path.join(tut2017_dir, f"audio_{idx}")):  # Check if the audio folder already exists
                    self._unzip_file(zip_file_path, tut2017_dir)

            # Download and unzip metadata file if not already done
            meta_url = "https://zenodo.org/records/1040168/files/TUT-acoustic-scenes-2017-evaluation.meta.zip"
            meta_file_path = os.path.join(tut2017_dir, "meta.zip")
            if not os.path.exists(meta_file_path):  # Check if the meta file already exists
                self._download_file(meta_url, meta_file_path)
            if not os.path.exists(os.path.join(tut2017_dir, "TUT-acoustic-scenes-2017-evaluation", "meta.txt")):  # Check if the extracted meta.txt exists
                self._unzip_file(meta_file_path, tut2017_dir)

            # Process the dataset
            return self.process_dataset(tut2017_dir)

        except Exception as e:
            print(f"Error downloading TUT2017 dataset: {str(e)}")
            return False

    def _download_file(self, url: str, file_path: str) -> None:
        """Helper method to download a file from a URL."""
        print(f"Downloading {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            raise RuntimeError(f"Failed to download file from {url}")

    def _unzip_file(self, zip_path: str, extract_dir: str) -> None:
        """Helper method to unzip a file."""
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    def process_dataset(self, dataset_dir: str) -> bool:
        """Process the dataset and create JSONL file."""
        meta_file = os.path.join(dataset_dir, "TUT-acoustic-scenes-2017-evaluation", "meta.txt")
        metadata_path = os.path.join(self.output_dir, "TUT2017.jsonl")

        # Read meta.txt file
        with open(meta_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()

        metadata = []
        question = "Identify the acoustic scene in the audio.\nOptions:\n(A) beach\n(B) bus\n(C) cafe or restaurant\n(D) car\n(E) city center\n(F) forest path\n(G) grocery store\n(H) home\n(I) library\n(J) metro station\n(K) office\n(L) park\n(M) residential area\n(N) train\n(O) tram\n.Answer with the option's letter from the given choices directly and only give the best option."
        
        # Process each line in the meta.txt file
        for index, line in enumerate(lines):
            parts = line.strip().split()
            audio_path = os.path.join(dataset_dir, "TUT-acoustic-scenes-2017-evaluation", parts[0])  # audio file path from meta.txt
            answer = parts[1]  # corresponding label

            # Ensure audio file exists
            assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist"

            metadata.append({
                "index": index,
                "question": question,
                "audio_path": audio_path,
                "answer": answer,
                "subset": "acoustic_scene_classification",
                "task_type": "understanding"
            })
        
        # Write the metadata to JSONL file if it doesn't already exist
        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w', encoding="utf-8") as f:
                for item in metadata:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Completed processing TUT2017 dataset. Metadata saved to {metadata_path}")
        else:
            print(f"TUT2017 JSONL file already exists, skipping processing.")

        return True


class CochlsceneDownloader(DatasetDownloader):
    """Downloader for Cochlscene dataset."""

    def download(self) -> bool:
        # Ensure the directory exists
        cochlscene_dir = os.path.join(self.output_dir, "CochlScene")
        if not os.path.exists(cochlscene_dir):
            os.makedirs(cochlscene_dir, exist_ok=True)

        try:
            # Define URLs for audio files
            audio_urls = [
                "https://zenodo.org/records/7080122/files/CochlScene.z01",
                "https://zenodo.org/records/7080122/files/CochlScene.z02",
                "https://zenodo.org/records/7080122/files/CochlScene.z03",
                "https://zenodo.org/records/7080122/files/CochlScene.z04",
                "https://zenodo.org/records/7080122/files/CochlScene.z05",
                "https://zenodo.org/records/7080122/files/CochlScene.zip"
            ]
            
            # Download the files if not already downloaded
            for url in audio_urls:
                zip_file_path = os.path.join(cochlscene_dir, os.path.basename(url))
                if not os.path.exists(zip_file_path):  # Check if the zip part already exists
                    self._download_file(url, zip_file_path)

            # Merge and unzip the dataset if it hasn't been done
            full_zip_path = os.path.join(cochlscene_dir, "full.zip")
            if not os.path.exists(full_zip_path):
                self._merge_and_unzip(cochlscene_dir)
            
            # Check if both Test and Val directories exist
            test_dir = os.path.join(cochlscene_dir, "CochlScene", "Test")
            val_dir = os.path.join(cochlscene_dir, "CochlScene", "Val")
            if not os.path.exists(test_dir) or not os.path.exists(val_dir):
                # Use zipfile to properly handle the split archive
                self._unzip_split_archive(cochlscene_dir)
            
            # Process the dataset
            return self.process_dataset(cochlscene_dir)

        except Exception as e:
            print(f"Error downloading Cochlscene dataset: {str(e)}")
            return False

    def _download_file(self, url: str, file_path: str) -> None:
        """Helper method to download a file from a URL."""
        print(f"Downloading {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            raise RuntimeError(f"Failed to download file from {url}")

    def _merge_and_unzip(self, directory: str) -> None:
        """Merge the zip parts using `zip -s 0` and unzip the full zip file."""
        # The last part should be the .zip file
        zip_file = os.path.join(directory, "CochlScene.zip")
        
        # Check if all parts exist
        for i in range(1, 6):
            part_file = os.path.join(directory, f"CochlScene.z{i:02d}")
            if not os.path.exists(part_file):
                raise FileNotFoundError(f"Missing split archive part: {part_file}")
        
        # Merge the split zip files into a full zip archive using `zip -s 0`
        merge_cmd = f"zip -s 0 {os.path.join(directory, 'CochlScene.zip')} --out {os.path.join(directory, 'full.zip')}"
        subprocess.run(merge_cmd, shell=True, check=True)

        # Unzip the merged file
        unzip_cmd = f"unzip {os.path.join(directory, 'full.zip')} -d {directory}"
        subprocess.run(unzip_cmd, shell=True, check=True)

    def _unzip_split_archive(self, directory: str) -> None:
        """Handle the unzipping of a split archive by merging and extracting."""
        # If the merged zip is already done, skip merging
        if not os.path.exists(os.path.join(directory, "full.zip")):
            self._merge_and_unzip(directory)

    def process_dataset(self, dataset_dir: str) -> bool:
        """Process the dataset and create JSONL file."""
        metadata_path = os.path.join(self.output_dir, "CochlScene.jsonl")
        
        question = "Identify the acoustic scene in the audio.\nOptions:\n(A) bus\n(B) cafe\n(C) car\n(D) crowdedindoor\n(E) elevator\n(F) kitchen\n(G) park\n(H) residentialarea\n(I) restaurant\n(J) restroom\n(K) street\n(L) subway\n(M) subwaystation\n.Answer with the option's letter from the given choices directly and only give the best option."
        
        # Directories for Test and Val data
        subsets = ['Test', 'Val']
        
        metadata = []
        
        # Process Test and Val directories
        for subset in subsets:
            subset_dir = os.path.join(dataset_dir, 'CochlScene', subset)
            if os.path.exists(subset_dir):
                for root, _, files in os.walk(subset_dir):
                    for file in files:
                        if file.endswith(".wav"):
                            audio_path = os.path.join(root, file)
                            answer = os.path.basename(root)  # The folder name corresponds to the scene type
                            
                            metadata.append({
                                "index": len(metadata),
                                "question": question,
                                "audio_path": audio_path,
                                "answer": answer,
                                "subset": subset,
                                "task_type": "understanding"
                            })
        
        # Write the metadata to JSONL file if it doesn't already exist
        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w', encoding="utf-8") as f:
                for item in metadata:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Completed processing Cochlscene dataset. Metadata saved to {metadata_path}")
        else:
            print(f"Cochlscene JSONL file already exists, skipping processing.")

        return True



# Registry of all available dataset downloaders
DATASET_REGISTRY = {
    "VoiceBench": VoiceBenchDownloader,
    "OpenAudioBench": OpenAudioBenchDownloader,
    "LibriSpeech": LibrispeechDownloader,
    "WenetSpeech": WenetspeechDownloader,
    "Fleurs": FleursDownloader,
    "AISHELL-1": Aishell1Downloader,
    "AISHELL-2": Aishell2Downloader,
    "MMAU": MMAUTestMiniDownloader,
    "ClothoAQA": ClothoAQADownloader,
    "VocalSound": VocalsoundDownloader,
    "Nonspeech7k": Nonspeech7kDownloader,
    "MELD": MeldDownloader,
    "TUT2017": TUT2017Downloader,
    "CochlScene": CochlsceneDownloader,
}


def main():
    parser = argparse.ArgumentParser(description='Download benchmark audio datasets')
    parser.add_argument('--datasets', type=str, default='all',
                        help='Comma-separated list of datasets to download. Use "all" for all datasets.')
    parser.add_argument('--output-dir', type=str, default='./downloaded_datasets',
                        help='Directory to store downloaded datasets')
    args = parser.parse_args()

    # get the absolute path of output_dir
    args.output_dir = os.path.abspath(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.datasets.lower() == 'all':
        datasets_to_download = list(DATASET_REGISTRY.keys())
    else:
        datasets_to_download = [ds.strip() for ds in args.datasets.split(',')]
        # Validate dataset names
        for ds in datasets_to_download:
            if ds not in DATASET_REGISTRY:
                print(f"Warning: Unknown dataset '{ds}'. Skipping.")
                datasets_to_download.remove(ds)
    
    # Download and process each dataset
    results = {}
    for dataset_name in datasets_to_download:
        print(f"\n{'='*40}\nProcessing dataset: {dataset_name}\n{'='*40}")
        
        dataset_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        downloader_class = DATASET_REGISTRY[dataset_name]
        downloader = downloader_class(dataset_dir)
        
        success = downloader.run()
        results[dataset_name] = "Success" if success else "Failed"
    
    # Print summary
    print("\n\n" + "="*60)
    print("Download Summary:")
    print("="*60)
    for dataset, status in results.items():
        print(f"{dataset.ljust(20)}: {status}")


if __name__ == "__main__":
    main()