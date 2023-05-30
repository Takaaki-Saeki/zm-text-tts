import logging
import re
import warnings
from typing import Iterable, List, Optional, Union
from espnet2.text.abs_tokenizer import AbsTokenizer
import argparse
import pathlib
import tqdm
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
import unicodedata


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
        "bs_ba": "bs",
        "ca_es": "ca",
        "hr_hr": "hr",
        "da_dk": "da",
        "nl_nl": "nl",
        "en_us": "en-us",
        "fi_fi": "fi",
        "fr_fr": "fr-fr",
        "de_de": "de",
        "el_gr": "el",
        "hu_hu": "hu",
        "is_is": "is",
        "ga_ie": "ga",
        "it_it": "it",
        "mt_mt": "mt",
        "nb_no": "nb",
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
        "fa_ir": "fa",
        "tr_tr": "tr",
        "uz_uz": "tr",
        "af_za": "af",
        "am_et": "am",
        "om_et": "om",
        "sw_ke": "sw",
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
        "id_id": "id",
        "jv_id": "id",
        "ms_my": "ms",
        "mi_nz": "ms",
        "th_th": "ms",
        "vi_vn": "vi",
        "cmn_hans_cn": "cmn",
        "yue_hant_hk": "yue",
        "ja_jp": "ja",
        "ko_kr": "ko",
    }


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


def tsv(args):
    """Performing G2P conversion and making new tsv files."""

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

    in_list = []
    with open(args.in_path, "r") as fr:
        for line in fr:
            in_list.append(line.strip())
    out_path = args.in_path.parent / (args.in_path.stem + "_phn.tsv")
    
    out_list = []
    print(f"Generating new tsv file for {args.data_type}...")
    for line in tqdm.tqdm(in_list):
        line_list = line.strip().split("\t")
        if len(line_list) < 5:
            continue
        lang = line_list[2]
        if args.data_type == "mailabs":
            lcode = langtable_mailabs()[lang]
        elif args.data_type == "css10":
            lcode = langtable_css10()[lang]
        lcode = g2p_langtable()[lcode]
        text = line_list[4]
        if args.normalize:
            text = basic_normalizer(text)
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
    
    with open(out_path, "w") as fw:
        fw.write("\n".join(out_list))


def voxp(args):
    """Run phoneme conversion."""

    phoneme_tokenizers = {}
    for lang in langtable_voxp().keys():
        lcode = langtable_voxp()[lang]
        lcode = g2p_langtable()[lcode]
        phoneme_tokenizers[lcode] = PhonemeTokenizer(lcode)

    text_path = args.in_path
    lang = text_path.parent.stem
    out_path_phn = args.in_path.parent / "sentences_phn.txt"
    
    in_list = []
    out_list_phn = []
    with open(text_path, "r") as fr:
        for line in fr:
            in_list.append(line.strip())
    
    print(f"Generating new tsv file for {lang} in {args.data_type}...")
    for line in tqdm.tqdm(in_list):
        lcode = langtable_voxp()[lang]
        lcode = g2p_langtable()[lcode]
        if args.normalize:
            text = basic_normalizer(line)
        tokenizer = phoneme_tokenizers[lcode]
        phn_text = " ".join(tokenizer.text2tokens(text))
        out_list_phn.append(phn_text)
        
    with open(out_path_phn, "w") as fw:
        fw.write("\n".join(out_list_phn))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=pathlib.Path, help="Input .tsv or sentences.txt files to be processed")
    parser.add_argument("--data_type", type=str, choices=["mailabs", "css10", "voxp"])
    parser.add_argument("--normalize", action="store_true", help="Normalize text before G2P conversion")
    args = parser.parse_args()

    if args.data_type in ("css10", "mailabs"):
        tsv(args)
    else:
        voxp(args)