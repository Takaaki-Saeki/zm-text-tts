#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=1

# Options for silence trimming
fs=16000
nj=32

do_trimming=true
token_type=byte
use_css10=false
use_fleurs=false
use_mailabs=false
use_other_tts_data=false
mos_filtering=true
byte_len_filtering=true
lang_set=null
holdout_lang_set=null
lang_family=null
spk_set=null
n_train_utt=null
override_spk_set=null
use_only_byte_for_bphn=false
bphn_phn_infer=false

log "$0 $*"
. utils/parse_options.sh

opts_data=""
if [ ${use_css10} = true ]; then
    opts_data+=" --use_css10"
fi
if [ ${use_fleurs} = true ]; then
    opts_data+=" --use_fleurs"
fi
if [ ${use_mailabs} = true ]; then
    opts_data+=" --use_mailabs"
fi
if [ ${use_other_tts_data} = true ]; then
    opts_data+=" --use_other_tts_data"
fi
if [ ${mos_filtering} = true ]; then
    opts_data+=" --mos_filtering"
fi
if [ ${byte_len_filtering} = true ]; then
    opts_data+=" --byte_len_filtering"
fi
if [ ${lang_set} != null ]; then
    opts_data+=" --lang_set ${lang_set}"
fi
if [ ${holdout_lang_set} != null ]; then
    opts_data+=" --holdout_lang_set ${holdout_lang_set}"
fi
if [ ${lang_family} != null ]; then
    opts_data+=" --lang_family ${lang_family}"
fi
if [ ${spk_set} != null ]; then
    opts_data+=" --spk_set ${spk_set}"
fi
if [ ${n_train_utt} != null ]; then
    opts_data+=" --n_train_utt ${n_train_utt}"
fi
if [ ${override_spk_set} != null ]; then
    opts_data+=" --override_spk_set ${override_spk_set}"
fi
if [ ${use_only_byte_for_bphn} = true ]; then
    opts_data+=" --use_only_byte_for_bphn"
fi
if [ ${bphn_phn_infer} = true ]; then
    opts_data+=" --bphn_phn_infer"
fi


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${MASMULTTS}" ]; then
   log "Fill the value of 'LIBRITTS' of db.sh"
   exit 1
fi
db_root=${MASMULTTS}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: We assume the dataset has already been downloaded."
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.py"
    python3 local/data_prep.py \
    --db_dir ${db_root} \
    --token_type ${token_type} \
    ${opts_data}

    for setname in test dev train; do
        dst=data/${setname}
        utils/fix_data_dir.sh $dst || exit 1
        utils/validate_data_dir.sh --no-feats $dst || exit 1
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ "${do_trimming}" = true ]; then
    log "stage 1: scripts/audio/trim_silence.sh"
    for setname in test dev train; do
        scripts/audio/trim_silence.sh \
            --cmd "${train_cmd}" \
            --nj "${nj}" \
            --fs "${fs}" \
            --win_length 1024 \
            --shift_length 256 \
            --threshold 60 \
            --min_silence 0.01 \
            "data/${setname}" "data/${setname}/log"
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
