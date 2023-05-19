# Learning to Speak from Text: Zero-Shot Multilingual Text-to-Speech with Unsupervised Text Pretraining

## Data preparation

## Modifying configs

## Train
```
./run.sh --stage 1 --stop-stage 6
```

## Inference
```
./decode.sh --stage 7 --stop-stage 7 \
  --gpu_inference false \
  --vocoder_file "${path to hifigan_ckpt}" \
  --inference_tag decode_hifigan
```