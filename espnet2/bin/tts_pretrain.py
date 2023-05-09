#!/usr/bin/env python3
from espnet2.tasks.tts_pretrain import TTSPretrainTask


def get_parser():
    parser = TTSPretrainTask.get_parser()
    return parser


def main(cmd=None):
    """TTS pretraining

    Example:

        % python tts_train.py asr --print_config --optim adadelta
        % python tts_train.py --config conf/train_asr.yaml
    """
    TTSPretrainTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
