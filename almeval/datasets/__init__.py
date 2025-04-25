import inspect

from loguru import logger

from . import ds_asr, ds_mqa, ds_openqa, ds_refqa
from .base import AudioBaseDataset
from .ds_asr import ASRDataset
from .ds_mqa import AudioMQADataset
from .ds_openqa import AudioOpenQADataset
from .ds_refqa import AudioRefQADataset

# 定义要排除的类
EXCLUDED_CLASSES = [
    'AudioBaseDataset',  # 基类
    'ASRDataset',        # ASR基类
    'AudioMQADataset',   # MQA基类
    'AudioOpenQADataset',  # 开放QA基类
    'AudioRefQADataset',  # 参考QA基类
]


def get_subclasses(base_class, module):
    """Get all subclasses of base_class in module"""
    return [
        cls for name, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, base_class) and cls.__name__ not in EXCLUDED_CLASSES and cls.EXCLUDE is False
    ]


# Automatically get all dataset classes
ASR_DATASETS = get_subclasses(ASRDataset, ds_asr)
MQA_DATASETS = get_subclasses(AudioMQADataset, ds_mqa)
OPEN_QA = get_subclasses(AudioOpenQADataset, ds_openqa)
REF_QA = get_subclasses(AudioRefQADataset, ds_refqa)

# remove translation datasets
ALL_DATSETS = ASR_DATASETS + MQA_DATASETS + REF_QA + OPEN_QA
AUDIO_QA = [ds for ds in ALL_DATSETS if ds.INTERACTIVE == 'Audio-QA']
AUDIO_EVENT = [ds for ds in ALL_DATSETS if ds.AUDIO_TYPE == 'AudioEvent']
CLOSE_QA = MQA_DATASETS + REF_QA
ALL_DATASETS = {ds.DATASET_NAME: ds for ds in ALL_DATSETS}


def build_dataset(name=None) -> AudioBaseDataset:
    datasets = []
    if name == 'all' or name is None:
        datasets = [ALL_DATASETS[k]() for k in ALL_DATASETS]
    elif name == 'asr':
        datasets = [DS() for DS in ASR_DATASETS]
    elif name == 'mqa':
        datasets = [DS() for DS in MQA_DATASETS]
    elif name == 'refqa':
        datasets = [DS() for DS in REF_QA]
    elif name == 'open-qa':
        datasets = [DS() for DS in OPEN_QA]
    elif name == 'audio-qa':
        datasets = [DS() for DS in AUDIO_QA]
    elif name == 'audio-event':
        datasets = [DS() for DS in AUDIO_EVENT]
    elif name == 'close-qa':
        datasets = [DS() for DS in CLOSE_QA]

    if len(datasets) != 0:
        valid_datasets = [ds for ds in datasets if ds.ok]
        logger.info(
            f'Trying to build {len(valid_datasets)} datasets for {name}, {len(valid_datasets)} successfully built')
        return valid_datasets

    else:
        if name not in ALL_DATASETS:
            raise ValueError(
                f"Dataset {name} not found, all supported datasets: {ALL_DATASETS.keys()}, or 'all', 'asr', 'mqa', 'refqa', 'open-qa', 'audio-qa', 'audio-event', 'close-qa'")
        dataset = ALL_DATASETS[name]()
        if dataset.ok:
            logger.info(
                f'Trying to build dataset {name}, {dataset} successfully built')
            return dataset
        else:
            raise ValueError(f'Dataset {name} build failed')
