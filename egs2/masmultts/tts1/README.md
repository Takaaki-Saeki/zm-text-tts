# Multilingual TTS Training

## Data preparation
See [here](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/README.md) for details.

You can switch whether to use each corpus by modifying `run.sh`.
For example, if you are using VoxPopuli, set `use_css10=true`.

For reproducibility, we parepare the utterance ids used for train, dev, test sets [here](https://github.com/Takaaki-Saeki/zm-text-tts/tree/master/egs2/masmultts/uttids_paper).

## Specifying languages to be used
Set `lang_set.txt` in `run.sh` to determine the languages to be used for multilingual TTS training.
if you set `null`, then the model uses all the languages contained in the specified corpora.
The format of `lang_set.txt` is as follows.
```
de_de
fr_fr
fi_fi
...
```
Use the language codes defined [here](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/lang_table.md).

Set `lang_set_decode.txt` used in `decode.sh` to define the languages to be used to inference.
This means that languages included in `lang_set_decode.txt` but not present in `lang_set.txt` are **zero-shot languages**.

## Configuring training settings
- When overriding token_list, specify the token list file as `token_list_override=<path_to_token_list_file>`. When you build the token list from the dataset, set `null`.
- Specify `use_lid=true` to use the language ids. Or when using [lang2vec](https://github.com/antonisa/lang2vec)-based language vectors as language embedding, specify `use_lvector=true`, or false when not used. Note that these settings need to be consistent with the text-based pretraining.
- If using M_AILABS and `mos_filtering=true`, you can enable speech-quality-based filtering for data selection.

## Configuring other training setting and model architectures.
Edit `train.yaml` to modify the configurations of model architecture and training conditions.

Especially, confirm the following setting if you used the language-aware embedding layer and the text-based pretraining.
```yaml
tts_conf:
    langs: 15  # Number of your total languages + 1 
    use_adapter: True
    adapter_type: "residual"
    use_encoder_w_lid: True
freeze_param: [
"tts.encoder.adapter",
"tts.encoder.embed",
"tts.lid_emb",
]
init_param: [
"../tts_pretrain_1/exp/tts_train_byte/latest.pth:tts_pretrain.encoder:tts.encoder",
"../tts_pretrain_1/exp/tts_train_byte/latest.pth:tts_pretrain.lid_emb:tts.lid_emb",
]
```

## Running reprocessing and training

```
$ ./run.sh --stage 1 --stop-stage 6
```

## Inference
Ensure the `decode.sh` is consistent with `run.sh` except for `lang_set`.
Then run the inference with the following command.
```
$ ./decode.sh --stage 7 --stop-stage 7 \
  --gpu_inference false \
  --vocoder_file "${path to hifigan_ckpt}" \
  --inference_tag decode_hifigan
```

You can find our HiFi-GAN checkpoint trained on LibriTTS, VCTK, and CSS10 [here](https://drive.google.com/drive/folders/1pemypbNBYJPf_rT2pzcnVk7GjK8WtX4E?usp=sharing).