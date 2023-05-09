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
stop_stage=0

token_type=byte
use_css10=false
use_fleurs=false
use_mailabs=false
use_voxp=false
use_cv=false
use_cc100=false
use_paracrawl=false
byte_len_filtering=true
lang_set=null
few_sampling_langs=null

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
if [ ${use_voxp} = true ]; then
    opts_data+=" --use_voxp"
fi
if [ ${use_cv} = true ]; then
    opts_data+=" --use_cv"
fi
if [ ${use_paracrawl} = true ]; then
    opts_data+=" --use_paracrawl"
fi
if [ ${use_cc100} = true ]; then
    opts_data+=" --use_cc100"
fi
if [ ${byte_len_filtering} = true ]; then
    opts_data+=" --byte_len_filtering"
fi
if [ ${lang_set} != null ]; then
    opts_data+=" --lang_set ${lang_set}"
fi
if [ ${few_sampling_langs} != null ]; then
    opts_data+=" --few_sampling_langs ${few_sampling_langs}"
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
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
