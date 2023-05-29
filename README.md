# Learning to Speak from Text for Low-Resource TTS

Implementation for our paper ["Learning to Speak from Text: Zero-Shot Multilingual Text-to-Speech with Unsupervised Text Pretraining"](https://arxiv.org/abs/2301.12596) to appear in [IJCAI 2023](https://ijcai-23.org/).  
This repository is standalone but highly dependent on [ESPnet](https://github.com/espnet/espnet).

>**Abstract:**<br>
While neural text-to-speech (TTS) has achieved human-like natural synthetic speech, multilingual TTS systems are limited to resource-rich languages due to the need for paired text and studio-quality audio data. This paper proposes a method for zero-shot multilingual TTS using text-only data for the target language. The use of text-only data allows the development of TTS systems for low-resource languages for which only textual resources are available, making TTS accessible to thousands of languages. Inspired by the strong cross-lingual transferability of multilingual language models, our framework first performs masked language model pretraining with multilingual text-only data. Then we train this model with a paired data in a supervised manner, while freezing a language-aware embedding layer. This allows inference even for languages not included in the paired data but present in the text-only data. Evaluation results demonstrate highly intelligible zero-shot TTS with a character error rate of less than 12% for an unseen language. All experiments were conducted using public datasets and the implementation will be made available for reproducibility.

## Environment setup
```shell
$ cd tools
$ ./setup_anaconda.sh ${output-dir-name|default=venv} ${conda-env-name|default=root} ${python-version|default=none}
# e.g.
$ ./setup_anaconda.sh miniconda zmtts 3.8
```
Then install espent.
```shell
$ make TH_VERSION={pytorch-version} CUDA_VERSION=${cuda-version}
# e.g.
$ make TH_VERSION=1.10.1 CUDA_VERSION=11.3
```
You can also setup system Python environment.
For other options, refer to the [ESPnet installation](https://espnet.github.io/espnet/installation.html).

## Data preparation
1. We assume a root directory for various multilingual data. Let `root_path="/workspace/root"` be the root directory. Then, specify the path to the root dir for `MASMULTTS` in db.sh.
2. Download and place the training data for TTS. If you use [css10](https://github.com/Kyubyong/css10), please downsample it from 22.05k to 16k.
3. For each token type (bytes or IPA symbols), create a TSV file to compile the data for TTS. The data format of each TSV file is as follows.
```
utt_name<tab>path_to_wav_file<tab>lang_name<tab>speaker_name<tab>utternace_text
...
```
You can make the TSV file by ruinning `make_css10_tsv.py`.
4. Since runtime multilingual G2P is not implemented in ESPnet, IPA symbols must be dumped in advance. Replace the utterance_text in the TSV file with IPA symbols adding the suffix `_phn`.
5. Place text datasets for the unsupervised text pre-training. If you are using byte tokens, put a list of utterance texts in `voxp_text/lm_data/${lang}/sentences.txt`. If you use IPA, you need to dump IPA symbols to `voxp_text/lm_data/${lang}/sentences_phn.txt` first.
As a result, the root directory and the following files look like the following.
```
- css10/
  |- german
  |- spanish
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
Note that you can also use [M_AILABS](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/).
Please see [TTS data prep](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/egs2/masmultts/tts1/local/data_prep.py) and [Pretraining data prep](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/egs2/masmultts/tts_pretrain_1/local/data_prep.py) for details.
6. Add the path to the root directory in `db.sh` as `MASMULTTS=${path to root dir}`. 

## Unsupervised text pretraining
Please see [here](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/egs2/masmultts/tts_pretrain_1/README.md).

## TTS training and inference
Please see [here](https://github.com/Takaaki-Saeki/zm-text-tts/blob/master/egs2/masmultts/tts1/README.md).

## Work in progress
- [x] Providing implementation on the paper in this standalone repo.
- [x] Prepare a script to automate the data preparation pipeline.
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

