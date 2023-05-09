#!/usr/bin/env python3
#  2022, The University of Tokyo; Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys
import pathlib
import subprocess

import kaldiio
import numpy as np
import torch
from tqdm.contrib import tqdm
import git

from espnet2.fileio.sound_scp import SoundScpReader

def lang2code():
    return {
        "af_za": "afr", "ast_es": "ast", "bn_in": "bni", "ckb_iq": "ckb",
        "da_dk": "dan", "es_419": "spa", "fi_fi": "fin", "gl_es": "glg",
        "hi_in": "hin", "id_id": "ind", "ja_jp": "jpn", "kea_cv": "kea",
        "ko_kr": "kor", "ln_cd": "lin", "lv_lv": "lav", "mn_mn": "mon",
        "my_mm": "mya", "nso_za": "nso", "or_in": "ory", "pt_br": "por",
        "sk_sk": "slk", "sr_rs": "srp", "te_in": "tel", "uk_ua": "ukr",
        "vi_vn": "vie", "yue_hant_hk": "yue", "am_et": "amh", "az_az": "azb",
        "bs_ba": "bos", "cmn_hans_cn": "cmn", "de_de": "deu", "et_ee": "ekk",
        "fil_ph": "fil", "gu_in": "guj", "hr_hr": "hrv", "ig_ng": "ibo",
        "jv_id": "jav", "kk_kz": "kaz", "ky_kg": "kir", "lo_la": "lao",
        "mi_nz": "mri", "mr_in": "mar", "nb_no": "nob", "ny_mw": "nya",
        "pa_in": "pan", "ro_ro": "ron", "sl_si": "slv", "sv_se": "swe",
        "tg_tj": "tgk", "umb_ao": "umb", "wo_sn": "wol", "zu_za": "zul",
        "ar_eg": "arz", "be_by": "bel", "ca_es": "cat", "cs_cz": "ces",
        "el_gr": "ell", "fa_ir": "pes", "fr_fr": "fra", "ha_ng": "hau",
        "hu_hu": "hun", "is_is": "isl", "ka_ge": "kat", "km_kh": "khm",
        "lb_lu": "ltz", "lt_lt": "lit", "mk_mk": "mkd", "ms_my": "zlm",
        "ne_np": "nep", "oc_fr": "oci", "pl_pl": "pol", "ru_ru": "rus",
        "sn_zw": "sna", "sw_ke": "swa", "th_th": "tha", "ur_pk": "urd",
        "xh_za": "xho", "as_in": "asm", "bg_bg": "bul", "ceb_ph": "ceb",
        "cy_gb": "cym", "en_us": "eng", "en_uk": "eng", "ff_sn": "ful",
        "ga_ie": "gle", "he_il": "heb", "hy_am": "hye", "it_it": "ita",
        "kam_ke": "kam", "kn_in": "kan", "lg_ug": "lug", "luo_ke": "luo",
        "ml_in": "mal", "mt_mt": "mlt", "nl_nl": "nld", "om_et": "orc",
        "ps_af": "pus", "sd_in": "snd", "so_so": "som", "ta_in": "tam",
        "tr_tr": "tur", "uz_uz": "uzb", "yo_ng": "yor"}


def get_parser():
    """Construct the parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--toolkit",
        type=str,
        help="Toolkit for Extracting language vectors.",
        choices=["lang2vec"],
    )
    parser.add_argument(
        "--feats_type",
        type=str,
        help="Feature type for Extracting language vectors.",
        choices=["geo", "fam", "learned", "inventory_knn", "syntax_knn", "phonology_knn"],
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level.")
    parser.add_argument(
        "--in_folder", type=pathlib.Path, help="Path to the input kaldi data directory."
    )
    parser.add_argument(
        "--out_folder",
        type=pathlib.Path,
        help="Output folder to save the xvectors.",
    )
    return parser


def main(argv):
    """Load the model, generate kernel and bandpass plots."""
    parser = get_parser()
    args = parser.parse_args(argv)

    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    if args.toolkit == "lang2vec":
        import lang2vec.lang2vec as l2v

        # Prepare spk2utt for mean x-vector
        utt2lang = dict()
        with open(args.in_folder / "utt2lang", "r") as fr:
            for line in fr:
                details = line.strip().split()
                utt2lang[details[0]] = details[1]
        langs = list(set(utt2lang.values()))

        lang2vec = {}
        for lang in tqdm(langs):
            lcode = lang2code()[lang]
            out = l2v.get_features(lcode, args.feats_type)
            lang2vec[lang] = np.asarray(out[lcode])

        os.makedirs(args.out_folder, exist_ok=True)
        writer_utt = kaldiio.WriteHelper(
            "ark,scp:{0}/lvector.ark,{0}/lvector.scp".format(args.out_folder)
        )
        writer_lang = kaldiio.WriteHelper(
            "ark,scp:{0}/lang_lvector.ark,{0}/lang_lvector.scp".format(args.out_folder)
        )
        
        writer_lang[lang] = lang2vec[lang]
        for utt in utt2lang.keys():
            writer_utt[utt] = lang2vec[utt2lang[utt]]
        writer_utt.close()
        writer_lang.close()

if __name__ == "__main__":
    main(sys.argv[1:])