import torch
import torchaudio

from ..utils.misc import print_once


def patch_chatglm_model_init(original_init):
    def new_init(self, config, empty_init=True, device=None):
        print_once('Using patched chatglm model init')
        # ensure device is torch.device type
        if isinstance(device, str):
            device = torch.device(device)

        # ensure config.torch_dtype is torch.dtype type
        if isinstance(config.torch_dtype, str):
            config.torch_dtype = getattr(torch, config.torch_dtype)

        # call original init function
        original_init(self, config, empty_init=empty_init, device=device)

    return new_init


def patch_glm4_voice_update_model_kwargs_for_generation(
    outputs,
    model_kwargs,
    is_encoder_decoder=False,
    num_new_tokens=1,
):
    # modified the source code to support new version of transformers
    # see: https://huggingface.co/THUDM/glm-4-voice-9b/discussions/2
    print_once('Using patched glm4_voice update_model_kwargs_for_generation')
    # update past_key_values
    for possible_cache_name in ['past_key_values', 'mems', 'past_buckets_states', 'cache_params']:
        if hasattr(outputs, possible_cache_name):
            if possible_cache_name in ('past_buckets_states', 'mems'):
                cache_name = 'past_key_values'
            else:
                cache_name = possible_cache_name
            model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
            break

    # update attention mask
    if 'attention_mask' in model_kwargs:
        attention_mask = model_kwargs['attention_mask']
        model_kwargs['attention_mask'] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        )

    # update position ids
    if 'position_ids' in model_kwargs:
        position_ids = model_kwargs['position_ids']
        new_position_id = position_ids[..., -1:].clone()
        new_position_id += 1
        model_kwargs['position_ids'] = torch.cat(
            [position_ids, new_position_id], dim=-1
        )

    model_kwargs['is_first_forward'] = False

    if model_kwargs.get('use_cache', True) and 'cache_position' in model_kwargs:
        model_kwargs['cache_position'] = model_kwargs['cache_position'][-1:] + num_new_tokens

    return model_kwargs


def patch_baichuan_load_audio_waveform(self, uri, return_tensors=True, do_normalize=False):
    # for mmau-test-mini: https://huggingface.co/baichuan-inc/Baichuan-Audio-Instruct/discussions/1#67e27c55ad5e6f59d8561187
    print_once('Using patched baichuan load_audio_waveform')
    # sample_rate, num_frames, num_channels, bits_per_sample, encoding=PCM_S
    metadata = torchaudio.info(uri)
    # assert(metadata.num_channels <= 2), "acoustic file with {} channels.".format(metadata.num_channels)
    waveform_tensor, _ = torchaudio.load(uri, normalize=True)
    if self.config.sampling_rate != metadata.sample_rate:
        waveform_tensor = torchaudio.functional.resample(
            waveform_tensor, metadata.sample_rate, self.config.sampling_rate, lowpass_filter_width=128)

    # downmix to mono channel https://trac.ffmpeg.org/wiki/AudioChannelManipulation
    if metadata.num_channels > 1:
        waveform_tensor = torch.mean(waveform_tensor, dim=0, keepdim=True)

    # normalized to zero mean
    if do_normalize:
        waveform_tensor = self.zero_mean_unit_var_norm(waveform_tensor)

    if return_tensors:  # (channels, samples)
        return waveform_tensor
    else:
        return waveform_tensor.numpy()
