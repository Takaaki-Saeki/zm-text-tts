# Learning to Speak from Text for Low-Resource TTS

Implementation for our paper ["Learning to Speak from Text: Zero-Shot Multilingual Text-to-Speech with Unsupervised Text Pretraining"](https://arxiv.org/abs/2301.12596).
This repository is standalone but highly dependent on [ESPnet](https://github.com/espnet/espnet).

>**Abstract:**<br>
While neural text-to-speech (TTS) has achieved human-like natural synthetic speech, multilingual TTS systems are limited to resource-rich languages due to the need for paired text and studio-quality audio data. This paper proposes a method for zero-shot multilingual TTS using text-only data for the target language. The use of text-only data allows the development of TTS systems for low-resource languages for which only textual resources are available, making TTS accessible to thousands of languages. Inspired by the strong cross-lingual transferability of multilingual language models, our framework first performs masked language model pretraining with multilingual text-only data. Then we train this model with a paired data in a supervised manner, while freezing a language-aware embedding layer. This allows inference even for languages not included in the paired data but present in the text-only data. Evaluation results demonstrate highly intelligible zero-shot TTS with a character error rate of less than 12% for an unseen language. All experiments were conducted using public datasets and the implementation will be made available for reproducibility.

## Environment setup
Refer to the [ESPnet installation](https://espnet.github.io/espnet/installation.html).

## Data preparation
1. We assume a root directory for various multilingual data. Let `root_path="/workspace/root"` be the root directory. Then, specify the path to the root dir for `MASMULTTS` in db.sh.
2. Download and place the training data for TTS. We assume [css10](https://github.com/Kyubyong/css10). Other datasets are also possible. 3.
3. For each token type (bytes or IPA symbols), create a TSV file to compile the data for TTS. The data format of each TSV file is as follows.
```
utt_name<tab>path_to_wav_file<tab>lang_name<tab>speaker_name<tab>utternace_text
...
```
For example, you can create the tsv files as
```
css10_de_achtgesichterambiwasse_0341    german/de/achtgesichterambiwasse/achtgesichterambiwasse_0341.wav        german  css10_de        leuchten perlenweiß die eirunden gepuderten mädchengesichter in jedem gemach 
css10_de_achtgesichterambiwasse_0342    german/de/achtgesichterambiwasse/achtgesichterambiwasse_0342.wav        german  css10_de        mal sitzen da dreißig in eisvogelblauen gewändern mit scharlachnen blumen bestickt mal dreißig in smaragdgrünen gewändern
...
```
If you are using IPA symbols, add the suffix `_phn`.
Since runtime multilingual G2P is not implemented in ESPnet, IPA symbols must be dumped in advance.

4. Place text datasets for the unsupervised text pre-training. If you are using byte tokens, put a list of utterance texts in `voxp_text/lm_data/${lang}/sentences.txt`. If you use IPA, you need to dump IPA symbols to `voxp_text/lm_data/${lang}/sentences_phn.txt` first.

As a result, the root directory and the following files will be in the following format.
```
- css10/
  |- dutch
  |- german
  ...
- voxp_text
  |- lm_data
     |- de
        |- sentences.txt
        |- sentences_phn.txt (if needed)
     |- es
      ...
- css10.tsv
- css10_phn.tsv (if needed)
```
Note that you can also use [M_AILABS](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/)).
Please see [TTS data prep](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/egs2/masmultts/tts1/local/data_prep.py) and [Pretraining data prep](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/egs2/masmultts/tts_pretrain_1/local/data_prep.py) for details.

## Unsupervised text pretraining
Please see [here](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/egs2/masmultts/tts_pretrain_1/README.md).

## TTS training and inference
Please see [here](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/egs2/masmultts/tts1/README.md).

## Work in progress
- [ ] Providing implementation on the paper in this standalone repo.
- [ ] Prepare a script to automate the data preparation pipeline.
- [ ] Integrating the implementation to ESPnet.
- [ ] Providing pretrained models through ESPnet.

## Citation
```bibtex
@article{saeki2023learning,
  title={Learning to Speak from Text: Zero-Shot Multilingual Text-to-Speech with Unsupervised Text Pretraining},
  author={Saeki, Takaaki and Maiti, Soumi and Li, Xinjian and Watanabe, Shinji and Takamichi, Shinnosuke and Saruwatari, Hiroshi},
  journal={arXiv preprint arXiv:2301.12596},
  year={2023}
}
```

