#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

################################## Configs to be set ##################################
token_type=byte                     # byte, phn, bphn
use_mailabs=false                   # whether to use m_ailabs dataset
use_css10=false                     # whether to use css10 dataset
use_voxp=true                       # whether to use voxp dataset
use_lid=true                        # whether to use language id
use_lvector=false                   # whether to use lang2vec-derived language vector
byte_len_filtering=true             # whether to filter out long sentences
lang_set=null                       # specifying languages to use
lang2lid_override=null              # overriding lang2lid mapping
token_list_override=null            # overriding token list
#######################################################################################

local_data_opts=""
local_data_opts+=" --token_type ${token_type}"
local_data_opts+=" --use_mailabs ${use_mailabs}"
local_data_opts+=" --use_css10 ${use_css10}"
local_data_opts+=" --use_fleurs ${use_fleurs}"
local_data_opts+=" --use_voxp ${use_voxp}"
local_data_opts+=" --use_cc100 ${use_cc100}"
local_data_opts+=" --byte_len_filtering ${byte_len_filtering}"
local_data_opts+=" --lang_set ${lang_set}"

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
elif [ ${token_type} = "phn" ]; then
    model_token_type=phn
    g2p=none
else
    echo "Error: token_type must be either byte or phn"
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
