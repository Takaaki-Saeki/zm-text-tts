#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

################# Configs to be set #####################
token_type=byte   # byte, tphn, phn, bphn
use_mailabs=true
use_css10=true
use_fleurs=true
use_voxp=true
use_cc100=true
use_lid=true
use_lvector=false
byte_len_filtering=true
lang_set=null
lang2lid_override=null
token_list_override=null
few_sampling_langs=null
#########################################################

local_data_opts=""
local_data_opts+=" --token_type ${token_type}"
local_data_opts+=" --use_mailabs ${use_mailabs}"
local_data_opts+=" --use_css10 ${use_css10}"
local_data_opts+=" --use_fleurs ${use_fleurs}"
local_data_opts+=" --use_voxp ${use_voxp}"
local_data_opts+=" --use_cc100 ${use_cc100}"
local_data_opts+=" --byte_len_filtering ${byte_len_filtering}"
local_data_opts+=" --lang_set ${lang_set}"
local_data_opts+=" --few_sampling_langs ${few_sampling_langs}"

opts=""
if [ ${lang2lid_override} != null ]; then
    opts+="--lang2lid_override ${lang2lid_override} "
fi
if [ ${token_list_override} != null ]; then
    opts+="--token_list_override ${token_list_override} "
fi

lang=noinfo
cleaner=none
if [ ${token_type} = "byte" ]; then
    model_token_type=byte
    g2p=byte
elif [ ${token_type} = "tphn" ]; then
    model_token_type=char
    g2p=none
elif [ ${token_type} = "phn" ]; then
    model_token_type=phn
    g2p=none
elif [ ${token_type} = "bphn" ]; then
    model_token_type=word
    g2p=none
else
    echo "Error: token_type must be either byte, tphn, phn, or bphn"
    exit 1
fi

train_config=conf/train.yaml

train_set=train
valid_set=dev
test_sets=test

./tts_pretrain.sh \
    --local_data_opts "${local_data_opts}" \
    --lang ${lang} \
    --use_lid ${use_lid} \
    --token_type "${model_token_type}" \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
