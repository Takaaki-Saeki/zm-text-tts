# Unsupervised text pretraining

## Data preparation
See [here](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/README.md) for details.

You can switch whether to use each corpus by modifying `run.sh`.
For example, if you are using VoxPopuli, set `use_voxp=true`.

## Specifying languages to be used
Set `lang_set.txt` in `run.sh` to determine the languages to be used for text-based pretraining.
if you set `null`, then the model uses all the languages contained in the specified corpora.
The format of `lang_set.txt` is as follows.
```
de_de
fr_fr
fi_fi
...
```
Use the language codes defined [here](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/lang_table.md).

## Configuring training settings
In `run.sh`, change the following settings to define the training conditions.
- When overriding token_list, specify the token list file as `token_list_override=<path_to_token_list_file>`. When you build the token list from the dataset, set `null`.
- Specify `use_lid=true` to use the language ids. Or when using [lang2vec](https://github.com/antonisa/lang2vec)-based language vectors as language embedding, specify `use_lvector=true`, or false when not used.

## Configuring of other training settings and model architectures
Edit `conf/train.yaml` to modify the configurations of model architecture and training conditions.
You may only need to change `tts_pretrain_conf: langs` based on your language set.

If you use the language-aware embedding layer, ensure the following config.
```yaml
tts_pretrain_conf:
    langs: 15                       # Number of your total languages + 1
    use_adapter: True               # whether to use language adapter
    adapter_type: "residual"        # type of adapter
```

## Running preprocessing and training
```
$ ./run.sh --stage 1 --stop-stage 5
```