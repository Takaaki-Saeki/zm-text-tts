import logging
import re
import warnings
from typing import Iterable, List, Optional, Union
from espnet2.text.abs_tokenizer import AbsTokenizer
import argparse
import codecs
import pathlib
import tqdm
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
import numpy
import os
import unicodedata

def tsv():
    """Run phoneme conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_text", type=pathlib.Path, help="Input kaldi-style text.")
    parser.add_argument("--out_text", type=pathlib.Path, help="Output kaldi-style text.")
    parser.add_argument("--data_type", type=str, choices=["mailabs", "css10", "fleurs", "voxp"])
    args = parser.parse_args()

    phoneme_tokenizers = {}
    if args.data_type == "mailabs":
        for lang in langtable_mailabs().keys():
            lcode = langtable_mailabs()[lang]
            lcode = g2p_langtable()[lcode]
            phoneme_tokenizers[lcode] = PhonemeTokenizer(lcode)
    elif args.data_type == "css10":
        for lang in langtable_css10().keys():
            lcode = langtable_css10()[lang]
            lcode = g2p_langtable()[lcode]
            phoneme_tokenizers[lcode] = PhonemeTokenizer(lcode)
    elif args.data_type == "fleurs":
        for lang in g2p_langtable().keys():
            lcode = g2p_langtable()[lang]
            phoneme_tokenizers[lcode] = PhonemeTokenizer(lcode)

    in_list = []
    with open(args.in_text, "r") as fr:
        for line in fr:
            in_list.append(line.strip())

    ignore_list = set([
        "hi_in_spk008_11633347711483585417"
    ])
    
    out_list = []
    for line in tqdm.tqdm(in_list):
        line_list = line.strip().split("\t")
        if len(line_list) < 5:
            continue
        lang = line_list[2]
        if args.data_type == "mailabs":
            lcode = langtable_mailabs()[lang]
        elif args.data_type == "css10":
            lcode = langtable_css10()[lang]
        elif args.data_type == "fleurs":
            lcode = lang
        uttid = line_list[0]
        if uttid in ignore_list:
            continue
        lcode = g2p_langtable()[lcode]
        text = line_list[4]
        tokenizer = phoneme_tokenizers[lcode]
        phn_text = " ".join(tokenizer.text2tokens(text))
        out_line = [
            line_list[0],
            line_list[1],
            line_list[2],
            line_list[3],
            phn_text
        ]
        out_list.append("\t".join(out_line))
    
    with open(args.out_text, "w") as fw:
        fw.write("\n".join(out_list))


def tsv2():
    """Run phoneme conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_text", type=pathlib.Path, help="Input kaldi-style text.")
    parser.add_argument("--data_type", type=str, choices=["mailabs", "css10", "fleurs"])
    args = parser.parse_args()

    phoneme_tokenizers = {}
    if args.data_type == "mailabs":
        data_name = "m_ailabs"
        for lang in langtable_mailabs().keys():
            lcode = langtable_mailabs()[lang]
            lcode = g2p_langtable()[lcode]
            phoneme_tokenizers[lcode] = PhonemeTokenizer(lcode)
    elif args.data_type == "css10":
        data_name = "css10"
        for lang in langtable_css10().keys():
            lcode = langtable_css10()[lang]
            lcode = g2p_langtable()[lcode]
            phoneme_tokenizers[lcode] = PhonemeTokenizer(lcode)
    elif args.data_type == "fleurs":
        data_name = "fleurs"
        for lang in g2p_langtable().keys():
            lcode = g2p_langtable()[lang]
            phoneme_tokenizers[lcode] = PhonemeTokenizer(lcode)

    in_list = []
    with open(args.in_text, "r") as fr:
        for line in fr:
            in_list.append(line.strip())

    ignore_list = set([
        "hi_in_spk008_11633347711483585417"
    ])
    
    out_lists = {
        "byte": [],
        "phn": [],
        "bphn": []
    }
    for line in tqdm.tqdm(in_list):
        line_list = line.strip().split("\t")
        if len(line_list) < 5:
            continue
        lang = line_list[2]
        if args.data_type == "mailabs":
            lcode = langtable_mailabs()[lang]
        elif args.data_type == "css10":
            lcode = langtable_css10()[lang]
        elif args.data_type == "fleurs":
            lcode = lang
        uttid = line_list[0]
        if uttid in ignore_list:
            continue
        lcode = g2p_langtable()[lcode]
        text = line_list[4]
        tokenizer = phoneme_tokenizers[lcode]
        phn_text = " ".join(tokenizer.text2tokens(text))
        byte_text = " ".join([str(x) for x in list(text.encode("utf-8"))])
        out_line_phn = [
            line_list[0],
            line_list[1],
            line_list[2],
            line_list[3],
            phn_text
        ]
        out_line_phn_bphn = [
            line_list[0]+"_phn",
            line_list[1],
            line_list[2],
            line_list[3],
            phn_text
        ]
        out_line_byte = [
            line_list[0],
            line_list[1],
            line_list[2],
            line_list[3],
            byte_text
        ]
        out_line_byte_bphn = [
            line_list[0]+"_byte",
            line_list[1],
            line_list[2],
            line_list[3],
            byte_text
        ]
        out_lists["byte"].append("\t".join(out_line_byte))
        out_lists["phn"].append("\t".join(out_line_phn))
        out_lists["bphn"].append("\t".join(out_line_byte_bphn))
        out_lists["bphn"].append("\t".join(out_line_phn_bphn))
    
    out_paths = {}
    out_paths["byte"] = args.in_text.parent / f"{data_name}_byte.tsv"
    out_paths["phn"] = args.in_text.parent / f"{data_name}_phn.tsv"
    out_paths["bphn"] = args.in_text.parent / f"{data_name}_bphn.tsv"
    
    for token in ["byte", "phn", "bphn"]:
        with open(out_paths[token], "w") as fw:
            fw.write("\n".join(out_lists[token]))


def voxp():
    """Run phoneme conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", type=pathlib.Path, help="Input kaldi-style text.")
    args = parser.parse_args()

    out_dir = pathlib.Path("data")

    phoneme_tokenizers = {}
    for lang in langtable_voxp().keys():
        lcode = langtable_voxp()[lang]
        lcode = g2p_langtable()[lcode]
        phoneme_tokenizers[lcode] = PhonemeTokenizer(lcode)

    for lang in langtable_voxp().keys():
        print(f"Processing {lang} ...")
        text_path = args.db_dir / lang / "sentences.txt"
        os.makedirs(out_dir / lang, exist_ok=True)
        out_path_phn = out_dir / lang / "sentences_phn.txt"
        out_path_byte = out_dir / lang / "sentences_byte.txt"
        out_path_bphn = out_dir / lang / "sentences_bphn.txt"
        if out_path_phn.exists():
            print(f"{lang} is already processed. Skipping.")
            continue
    
        in_list = []
        out_list_phn = []
        out_list_byte = []
        out_list_bphn = []
        with open(text_path, "r") as fr:
            for line in fr:
                in_list.append(line.strip())
            
        for line in tqdm.tqdm(in_list):
            lcode = langtable_voxp()[lang]
            lcode = g2p_langtable()[lcode]
            text = basic_normalizer(line)
            if len(text) == 0:
                continue
            try:
                tokenizer = phoneme_tokenizers[lcode]
                phn_text = " ".join(tokenizer.text2tokens(text))
                byte_text = " ".join([str(x) for x in list(text.encode("utf-8"))])
                out_list_phn.append(phn_text)
                out_list_byte.append(byte_text)
                out_list_bphn.append(phn_text)
                out_list_bphn.append(byte_text)
            except:
                continue
        
        with open(out_path_phn, "w") as fw:
            fw.write("\n".join(out_list_phn))
        with open(out_path_byte, "w") as fw:
            fw.write("\n".join(out_list_byte))
        with open(out_path_bphn, "w") as fw:
            fw.write("\n".join(out_list_bphn))

def cc100():
    """Run phoneme conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", type=pathlib.Path, help="Input kaldi-style text.")
    args = parser.parse_args()

    out_dir = pathlib.Path("data")

    phoneme_tokenizers = {}
    langs = [
        "af_za", "el_gr", "ru_ru", "ja_jp", "pl_pl","te_in", "gu_in",
        "hi_in", "bn_in", "ta_in", "uk_ua", "yo_ng", "gl_es", "xh_za",
        "km_kh", "my_mm", "pa_in", "jv_id", "ml_in", "ne_np"]
    for lang in langs:
        lcode = g2p_langtable()[lang]
        phoneme_tokenizers[lcode] = PhonemeTokenizer(lcode)

    for lang in langs:
        print(f"Processing {lang} ...")
        text_path = args.db_dir / f"{lang}.txt"
        os.makedirs(out_dir / lang, exist_ok=True)
        out_path_phn = out_dir / lang / "sentences_phn.txt"
        out_path_byte = out_dir / lang / "sentences_byte.txt"
        out_path_bphn = out_dir / lang / "sentences_bphn.txt"
        if out_path_phn.exists():
            print(f"{lang} is already processed. Skipping.")
            continue
    
        in_list = []
        out_list_phn = []
        out_list_byte = []
        out_list_bphn = []
        with open(text_path, "r") as fr:
            for line in fr:
                in_list.append(line.strip())
            
        for line in tqdm.tqdm(in_list):
            lcode = g2p_langtable()[lang]
            text = basic_normalizer(line)
            if len(text) == 0:
                continue
            try:
                tokenizer = phoneme_tokenizers[lcode]
                phn_text = " ".join(tokenizer.text2tokens(text))
                byte_text = " ".join([str(x) for x in list(text.encode("utf-8"))])
                out_list_phn.append(phn_text)
                out_list_byte.append(byte_text)
                out_list_bphn.append(phn_text)
                out_list_bphn.append(byte_text)
            except:
                continue
        
        with open(out_path_phn, "w") as fw:
            fw.write("\n".join(out_list_phn))
        with open(out_path_byte, "w") as fw:
            fw.write("\n".join(out_list_byte))
        with open(out_path_bphn, "w") as fw:
            fw.write("\n".join(out_list_bphn))

def remove_symbols(s: str):
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s)
    )

def basic_normalizer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
    s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
    s = remove_symbols(s).lower()
    s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space
    return s


class Phonemizer:
    """Phonemizer module for various languages.
    This is wrapper module of https://github.com/bootphon/phonemizer.
    You can define various g2p modules by specifying options for phonemizer.
    See available options:
        https://github.com/bootphon/phonemizer/blob/master/phonemizer/phonemize.py#L32
    """

    def __init__(
        self,
        backend,
        word_separator: Optional[str] = None,
        syllable_separator: Optional[str] = None,
        phone_separator: Optional[str] = " ",
        strip=False,
        split_by_single_token: bool = False,
        **phonemizer_kwargs,
    ):
        # delayed import
        from phonemizer.backend import BACKENDS
        from phonemizer.separator import Separator

        self.separator = Separator(
            word=word_separator,
            syllable=syllable_separator,
            phone=phone_separator,
        )

        # define logger to suppress the warning in phonemizer
        logger = logging.getLogger("phonemizer")
        logger.setLevel(logging.ERROR)
        self.phonemizer = BACKENDS[backend](
            **phonemizer_kwargs,
            logger=logger,
        )
        self.strip = strip
        self.split_by_single_token = split_by_single_token

    def __call__(self, text) -> List[str]:
        tokens = self.phonemizer.phonemize(
            [text],
            separator=self.separator,
            strip=self.strip,
            njobs=1,
        )[0]
        if not self.split_by_single_token:
            return tokens.split()
        else:
            # "a: ab" -> ["a", ":", "<space>",  "a", "b"]
            # TODO(kan-bayashi): space replacement should be dealt in PhonemeTokenizer
            return [c.replace(" ", "<space>") for c in tokens]


class PhonemeTokenizer(AbsTokenizer):
    def __init__(
        self,
        lang,
        non_linguistic_symbols: Union[pathlib.Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False
    ):
        self.g2p = Phonemizer(
            language=lang,
            backend="espeak",
            with_stress=False,
            preserve_punctuation=False
        )
        self.space_symbol = space_symbol
        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (pathlib.Path, str)):
            non_linguistic_symbols = pathlib.Path(non_linguistic_symbols)
            try:
                with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                    self.non_linguistic_symbols = set(line.rstrip() for line in f)
            except FileNotFoundError:
                warnings.warn(f"{non_linguistic_symbols} doesn't exist.")
                self.non_linguistic_symbols = set()
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'g2p_type="{self.g2p_type}", '
            f'space_symbol="{self.space_symbol}", '
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            ")"
        )

    def text2tokens(self, line: str) -> List[str]:
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols:
                if line.startswith(w):
                    if not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(w)])
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                tokens.append(t)
                line = line[1:]

        line = "".join(tokens)
        tokens = self.g2p(line)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        # phoneme type is not invertible
        return "".join(tokens)


def langtable_mailabs():
    return {
        "de_DE": "de_de",
        "en_US": "en_us",
        "en_UK": "en_uk",
        "es_ES": "es_419",
        "fr_FR": "fr_fr",
        "it_IT": "it_it",
        "pl_PL": "pl_pl",
        "ru_RU": "ru_ru",
        "uk_UK": "uk_ua",
    }

def langtable_css10():
    return {
        "chinese": "cmn_hans_cn",
        "dutch": "nl_nl",
        "finnish": "fi_fi",
        "french": "fr_fr",
        "german": "de_de",
        "greek": "el_gr",
        "hungarian": "hu_hu",
        "japanese": "ja_jp",
        "russian": "ru_ru",
        "spanish": "es_419",
    }

def langtable_voxp():
    return {
        "en": "en_us",
        "de": "de_de",
        "es": "es_419",
        "cs": "cs_cz",
        "fi": "fi_fi",
        "fr": "fr_fr",
        "hr": "hr_hr",
        "hu": "hu_hu",
        "it": "it_it",
        "nl": "nl_nl",
        "pl": "pl_pl",
        "ro": "ro_ro",
        "sk": "sk_sk",
        "sl": "sl_si",
        "et": "et_ee",
        "lt": "lt_lt"
    }

def g2p_langtable():
    return {
        "ast_es": "es", # Not existed
        "bs_ba": "bs",
        "ca_es": "ca",
        "hr_hr": "hr",
        "da_dk": "da",
        "nl_nl": "nl",
        "en_uk": "en-us",
        "en_us": "en-us",
        "fi_fi": "fi",
        "fr_fr": "fr-fr",
        "gl_es": "pt", # Not existed
        "de_de": "de",
        "el_gr": "el",
        "hu_hu": "hu",
        "is_is": "is",
        "ga_ie": "ga",
        "it_it": "it",
        "kea_cv": "pt", # Not existed
        "lb_lu": "pt", # Not existed
        "mt_mt": "mt",
        "nb_no": "nb",
        "oc_fr": "ca", # Not existed
        "pt_br": "pt-br",
        "es_419": "es",
        "sv_se": "sv",
        "cy_gb": "cy",
        "hy_am": "hy",
        "be_by": "ru",
        "bg_bg": "bg",
        "cs_cz": "cs",
        "et_ee": "et",
        "ka_ge": "ka",
        "lv_lv": "lv",
        "lt_lt": "lt",
        "mk_mk": "mk",
        "pl_pl": "pl",
        "ro_ro": "ro",
        "ru_ru": "ru",
        "sr_rs": "sr",
        "sk_sk": "sk",
        "sl_si": "sl",
        "uk_ua": "ru",
        "ar_eg": "ar",
        "az_az": "az",
        "he_il": "ar",
        "kk_kz": "ar",
        "ky_kg": "ky",
        "mn_mn": "ar", # Not existed
        "ps_af": "fa", # Not existed
        "fa_ir": "fa",
        "ckb_iq": "ku", # Not existed
        "tg_tj": "fa", # Not existed
        "tr_tr": "tr",
        "uz_uz": "tr",
        "af_za": "af",
        "am_et": "am",
        "ff_sn": "om", # Not existed
        "lg_ug": "om", # Not existed
        "ha_ng": "om", # Not existed
        "ig_ng": "om", # Not existed
        "kam_ke": "om", # Not existed
        "ln_cd": "om", # Not existed 
        "luo_ke": "om", # Not existed
        "nso_za": "om", # Not existed
        "ny_mw": "om", # Not existed
        "om_et": "om",
        "sn_zw": "sw", # Not existed
        "so_so": "sw", # Not existed
        "sw_ke": "sw",
        "umb_ao": "sw", # Not existed
        "wo_sn": "sw", # Not existed
        "xh_za": "sw", # Not existed
        "yo_ng": "sw", # Not existed
        "zu_za": "sw", # Not existed
        "as_in": "as",
        "bn_in": "bn",
        "gu_in": "gu",
        "hi_in": "hi",
        "kn_in": "kn",
        "ml_in": "ml",
        "mr_in": "mr",
        "ne_np": "ne",
        "or_in": "or",
        "pa_in": "pa",
        "sd_in": "sd",
        "ta_in": "ta",
        "te_in": "te",
        "ur_pk": "ur",
        "my_mm": "my",
        "ceb_ph": "my", # Not existed
        "fil_ph": "my", # Not existed
        "id_id": "id",
        "jv_id": "id",
        "km_kh": "ms", # Not existed
        "lo_la": "ms", # Not existed
        "ms_my": "ms",
        "mi_nz": "ms",
        "th_th": "ms",
        "vi_vn": "vi",
        "cmn_hans_cn": "cmn",
        "yue_hant_hk": "yue",
        "ja_jp": "ja",
        "ko_kr": "ko",
    }


if __name__ == "__main__":
    tsv2()
    #cc100()
    #voxp()