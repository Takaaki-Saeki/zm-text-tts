#!/usr/bin/env bash

# Copyright 2022 Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
    local a b
    a=$1
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a="${b}"
        fi
    done
    echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true  # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts="" # Options to be passed to local/data.sh.

# Feature extraction related
use_lid=false # Whether to use language id as the inputs (Need utt2lang in data directory).

# Vocabulary related
oov="<unk>"         # Out of vocabrary symbol.
blank="<blank>"     # CTC blank symbol.
sos_eos="<sos/eos>" # sos and eos symbols.

# Training related
train_config="" # Config for training.
train_args=""   # Arguments for training, e.g., "--max_epoch 1".
# Note that it will overwrite args in train config.
tag=""                # Suffix for training directory.
tts_exp=""            # Specify the directory path for experiment. If this option is specified, tag is ignored.
tts_stats_dir=""      # Specify the directory path for statistics. If empty, automatically decided.
num_splits=1          # Number of splitting for tts corpus.
tts_task=tts_pretrain # TTS task (tts or gan_tts).

# [Task dependent] Set the datadir name created by local/data.sh
train_set=""         # Name of training set.
valid_set=""         # Name of validation set used for monitoring/tuning network training.
test_sets=""         # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
srctexts=""          # Texts to create token list. Multiple items can be specified.
nlsyms_txt=none      # Non-linguistic symbol list (needed if existing).
token_type=phn       # Transcription type (char or phn).
cleaner=tacotron     # Text cleaner.
g2p=g2p_en           # g2p method (needed if token_type=phn).
lang=noinfo          # The language type of corpus.
text_fold_length=150 # fold_length for text data.
lang2lid_override=null # Whether to override lang2lid.
token_list_override=null # Whether to override token_list.

# Upload model related
hf_repo=

help_message=$(
    cat <<EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>" --srctexts "<srctexts>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data prep related
    --local_data_opts # Options to be passed to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --use_lid          # Whether to use language id as the inputs (default="${use_lid}").
    --oov              # Out of vocabrary symbol (default="${oov}").
    --blank            # CTC blank symbol (default="${blank}").
    --sos_eos          # sos and eos symbole (default="${sos_eos}").

    # Training related
    --train_config  # Config for training (default="${train_config}").
    --train_args    # Arguments for training (default="${train_args}").
                    # e.g., --train_args "--max_epoch 1"
                    # Note that it will overwrite args in train config.
    --tag           # Suffix for training directory (default="${tag}").
    --tts_exp       # Specify the directory path for experiment.
                    # If this option is specified, tag is ignored (default="${tts_exp}").
    --tts_stats_dir # Specify the directory path for statistics.
                    # If empty, automatically decided (default="${tts_stats_dir}").
    --num_splits    # Number of splitting for tts corpus (default="${num_splits}").
    --tts_task              # TTS task {tts or gan_tts} (default="${tts_task}").

    # [Task dependent] Set the datadir name created by local/data.sh.
    --train_set          # Name of training set (required).
    --valid_set          # Name of validation set used for monitoring/tuning network training (required).
    --test_sets          # Names of test sets (required).
                         # Note that multiple items (e.g., both dev and eval sets) can be specified.
    --srctexts           # Texts to create token list (required).
                         # Note that multiple items can be specified.
    --nlsyms_txt         # Non-linguistic symbol list (default="${nlsyms_txt}").
    --token_type         # Transcription type (default="${token_type}").
    --cleaner            # Text cleaner (default="${cleaner}").
    --g2p                # g2p method (default="${g2p}").
    --lang               # The language type of corpus (default="${lang}").
    --text_fold_length   # Fold length for text data (default="${text_fold_length}").
    --lang2lid_override  # Whether to override lang2lid (default="${lang2lid_override}").
    --token_list_override # Whether to override token_list (default="${token_list_override}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# Check feature type
data_feats="${dumpdir}/text"

# Check token list type
token_listdir="${dumpdir}/token_list/${token_type}"
if [ "${cleaner}" != none ]; then
    token_listdir+="_${cleaner}"
fi
if [ "${token_type}" = phn ]; then
    token_listdir+="_${g2p}"
fi
token_list="${token_listdir}/tokens.txt"

# Check old version token list dir existence
if [ -e data/token_list ] && [ ! -e "${dumpdir}/token_list" ]; then
    log "Default token_list directory path is changed from data to ${dumpdir}."
    log "Copy data/token_list to ${dumpdir}/token_list for the compatibility."
    [ ! -e ${dumpdir} ] && mkdir -p ${dumpdir}
    cp -a "data/token_list" "${dumpdir}/token_list"
fi

# Set tag for naming of model directory
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="$(basename "${train_config}" .yaml)_${token_type}"
    else
        tag="train_${token_type}"
    fi
    if [ "${cleaner}" != none ]; then
        tag+="_${cleaner}"
    fi
    if [ "${token_type}" = phn ]; then
        tag+="_${g2p}"
    fi
    # Add overwritten arg's info
    if [ -n "${train_args}" ]; then
        tag+="$(echo "${train_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${tts_stats_dir}" ]; then
    tts_stats_dir="${expdir}/tts_stats"
    tts_stats_dir+="_${token_type}"
    if [ "${cleaner}" != none ]; then
        tts_stats_dir+="_${cleaner}"
    fi
    if [ "${token_type}" = phn ]; then
        tts_stats_dir+="_${g2p}"
    fi
fi
# The directory used for training commands
if [ -z "${tts_exp}" ]; then
    tts_exp="${expdir}/tts_${tag}"
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        # TODO(kamo): Change kaldi-ark to npy or HDF5?
        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        # Prepare lang id input
        if "${use_lid}"; then
            log "Stage 2+: Prepare lang id: data/ -> ${data_feats}/"
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                mkdir -p "${data_feats}${_suf}"
                cp -r data/"${dset}" "${data_feats}${_suf}/"
                if [ "${dset}" = "${train_set}" ]; then
                    # Make lang2lid
                    # NOTE(kan-bayashi): 0 is reserved for unknown languages
                    echo "<unk> 0" >"${data_feats}${_suf}/${dset}/lang2lid"
                    cut -f 2 -d " " "${data_feats}${_suf}/${dset}/utt2lang" | sort | uniq |
                        awk '{print $1 " " NR}' >>"${data_feats}${_suf}/${dset}/lang2lid"
                fi
                if [ "${lang2lid_override}" != null ]; then
                    # Override lang2lid
                    # NOTE(Takaaki-Saeki): Overriding language id for pretraining
                    log "Overriding lang2lid with ${lang2lid_override}"
                    rm -rf "${data_feats}${_suf}/${dset}/lang2lid"
                    cp "${lang2lid_override}" "${data_feats}${_suf}/${dset}/lang2lid"
                fi
                # NOTE(kan-bayashi): We can reuse the same script for making utt2sid
                pyscripts/utils/utt2spk_to_utt2sid.py \
                    "${data_feats}/org/${train_set}/lang2lid" \
                    "${data_feats}${_suf}/${dset}/utt2lang" \
                    >"${data_feats}${_suf}/${dset}/utt2lid"
            done

            # Moving data
            for dset in "${train_set}" "${valid_set}"; do
                # Copy data dir
                cp -r "${data_feats}/org/${dset}" "${data_feats}/"
                if [ -e "${data_feats}/org/${dset}/utt2lid" ]; then
                    cp "${data_feats}/org/${dset}/utt2lid" "${data_feats}/${dset}/utt2lid"
                fi
            done
        else
            # Only copying data
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                mkdir -p "${data_feats}${_suf}"
                cp -r data/"${dset}" "${data_feats}${_suf}/"
            done
            for dset in "${train_set}" "${valid_set}"; do
                # Copy data dir
                cp -r "${data_feats}/org/${dset}" "${data_feats}/"
            done
        fi
    fi

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Generate token_list from ${srctexts}"
        # "nlsyms_txt" should be generated by local/data.sh if need

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task

        # shellcheck disable=SC2002
        cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"

        ${python} -m espnet2.bin.tokenize_text \
            --token_type "${token_type}" -f 2- \
            --input "${data_feats}/srctexts" --output "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --write_vocabulary true \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"
    fi
    if [ ${token_list_override} != null ]; then
        log "Overwrite the token_list with ${token_list_override}"
        rm -rf "${token_list}"
        cp "${token_list_override}" "${token_list}"
    fi
else
    log "Skip the stages for data preparation"
fi

# ========================== Data preparation is done here. ==========================

if ! "${skip_train}"; then
    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 4: TTS Pretraining collect stats: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        if "${use_lid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2lid,lids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2lid,lids,text_int "
        fi

        # 1. Split the key file
        _logdir="${tts_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(wc <${_train_dir}/text -l)" "$(wc <${_valid_dir}/text -l)")

        key_file="${_train_dir}/text"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_valid_dir}/text"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${tts_stats_dir}/run.sh'. You can resume the process from stage 4 using this script"
        mkdir -p "${tts_stats_dir}"
        echo "${run_args} --stage 4 \"\$@\"; exit \$?" >"${tts_stats_dir}/run.sh"
        chmod +x "${tts_stats_dir}/run.sh"

        # 3. Submit jobs
        log "TTS Pretraining collect_stats started... log: '${_logdir}/stats.*.log'"
        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m "espnet2.bin.${tts_task}" \
            --collect_stats true \
            --use_preprocessor true \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --train_data_path_and_name_and_type "${_train_dir}/text,text,text" \
            --valid_data_path_and_name_and_type "${_valid_dir}/text,text,text" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${train_args} || {
            cat $(grep -l -i error "${_logdir}"/stats.*.log)
            exit 1
        }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${tts_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        awk <"${tts_stats_dir}/train/text_shape" \
            -v N="$(wc <${token_list} -l)" '{ print $0 "," N }' \
            >"${tts_stats_dir}/train/text_shape.${token_type}"

        awk <"${tts_stats_dir}/valid/text_shape" \
            -v N="$(wc <${token_list} -l)" '{ print $0 "," N }' \
            >"${tts_stats_dir}/valid/text_shape.${token_type}"
    fi

    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: TTS Preraining: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        if [ "${num_splits}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${tts_stats_dir}/splits${num_splits}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                    --scps \
                    "${_train_dir}/text" \
                    "${_train_dir}/${_scp}" \
                    "${tts_stats_dir}/train/text_shape.${token_type}" \
                    --num_splits "${num_splits}" \
                    --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
            _opts+="--train_shape_file ${tts_stats_dir}/train/text_shape.${token_type} "
        fi
        _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
        _opts+="--valid_shape_file ${tts_stats_dir}/valid/text_shape.${token_type} "

        # Add language ID to the inputs if needed
        if "${use_lid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2lid,lids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2lid,lids,text_int "
        fi

        log "Generate '${tts_exp}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${tts_exp}"
        echo "${run_args} --stage 5 \"\$@\"; exit \$?" >"${tts_exp}/run.sh"
        chmod +x "${tts_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

        log "TTS Pretraining started... log: '${tts_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &>/dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${tts_exp})"
        else
            jobname="${tts_exp}/train.log"
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${tts_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${tts_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m "espnet2.bin.${tts_task}" \
            --use_preprocessor true \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --resume true \
            --fold_length "${text_fold_length}" \
            --output_dir "${tts_exp}" \
            ${_opts} ${train_args}

    fi
else
    log "Skip training stages"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
