from almeval.models import build_model

test_cases = [
        {
            "index": 0,
            "text": "请把这段语音转录成文本。",
            "audio": ["tests/data/asr_example.wav"],
            "meta": {
                "subset": "test",
                "audio_type": "Speech",
                "task": "ASR",
                "dataset_name": 'Fleurs-zh',
                'interactive': 'Audio-analysis',
                'dataset_series': 'Fleurs',
                'lang': 'zh',
            }
        }
    ]
def test_load_step_audio():
    # you need at least 80G x 4 to run this.
    model_name = "StepAudio"
    model = build_model(model_name)
    assert model is not None
    assert model.NAME == model_name
    for item in test_cases:
        res = model(item)
        assert res is not None
        print(res)


def test_load_models():
    all_models = ["Baichuan-Audio", "Baichuan-Audio-Chat", "Qwen2-Audio-7B", "Qwen2-Audio-7B-Instruct", "Qwen2.5-Omni-7B", "GLM4-Voice", "Kimi-Audio"]
    for model_name in all_models:
        model = build_model(model_name)
        assert model is not None
        assert model.NAME == model_name
        for item in test_cases:
            res = model(item)
            assert res is not None
            print(res)
