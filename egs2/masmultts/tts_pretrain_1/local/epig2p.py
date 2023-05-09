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
import epitran

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
    parser.add_argument("--data_type", type=str, choices=["other_tts_data"])
    args = parser.parse_args()

    epis = {}
    data_name = "other_tts_data"
    for lang in g2p_langtable().keys():
        lcode = g2p_langtable()[lang]
        epis[lcode] = epitran.Epitran(lcode)

    in_list = []
    with open(args.in_text, "r") as fr:
        for line in fr:
            in_list.append(line.strip())
    
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
        lcode = lang
        text = line_list[4]
        if lcode in g2p_langtable().keys():
            lcode = g2p_langtable()[lcode]
            tokenizer = epis[lcode]
            processed = [c for c in tokenizer.trans_list(text) if not c.isspace()]
            processed = [c if (not c.isnumeric()) else "<NUM>" for c in processed]
            phn_text = " ".join(processed)
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
            out_lists["phn"].append("\t".join(out_line_phn))
            out_lists["bphn"].append("\t".join(out_line_phn_bphn))

        byte_text = " ".join([str(x) for x in list(text.encode("utf-8"))])
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
        out_lists["bphn"].append("\t".join(out_line_byte_bphn))
    
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

        cnt = 0
            
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

    epis = {}
    langs = [
        "te_in", "hi_in", "bn_in", "ta_in",
        "uk_ua", "yo_ng", "xh_za", "pa_in",
        "jv_id", "ml_in"]
    for lang in langs:
        lcode = g2p_langtable()[lang]
        print(f"Loading {lcode} ...")
        epis[lcode] = epitran.Epitran(lcode)

    for lang in langs:
        print(f"Processing {lang} ...")
        text_path = args.db_dir / lang / "sentences.txt"
        os.makedirs(args.db_dir / lang, exist_ok=True)
        out_path_phn = args.db_dir / lang / "sentences_phn.txt"
        out_path_byte = args.db_dir / lang / "sentences_byte.txt"
        out_path_bphn = args.db_dir / lang / "sentences_bphn.txt"
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
                tokenizer = epis[lcode]
                processed = [c for c in tokenizer.trans_list(text) if not c.isspace()]
                processed = [c if (not c.isnumeric()) else "<NUM>" for c in processed]
                phn_text = " ".join(processed)
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

def cc100_non():
    """Run phoneme conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", type=pathlib.Path, help="Input kaldi-style text.")
    args = parser.parse_args()

    langs = ["gu_in"]
    for lang in langs:
        print(f"Processing {lang} ...")
        text_path = args.db_dir / lang / "sentences.txt"
        os.makedirs(args.db_dir / lang, exist_ok=True)
        out_path_byte = args.db_dir / lang / "sentences_byte.txt"
        if out_path_byte.exists():
            print(f"{lang} is already processed. Skipping.")
            continue
    
        in_list = []
        out_list_byte = []
        with open(text_path, "r") as fr:
            for line in fr:
                in_list.append(line.strip())
            
        for line in tqdm.tqdm(in_list):
            text = basic_normalizer(line)
            if len(text) == 0:
                continue
            try:
                byte_text = " ".join([str(x) for x in list(text.encode("utf-8"))])
                out_list_byte.append(byte_text)
            except:
                continue
        with open(out_path_byte, "w") as fw:
            fw.write("\n".join(out_list_byte))


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
        "en_us": "eng-Latn",
        "de_de": "deu-Latn",
        "es_419": "spa-Latn",
        "cs_cz": "ces-Latn",
        "fr_fr": "fra-Latn",
        "hr_hr": "hrv-Latn",
        "hu_hu": "hun-Latn",
        "it_it": "ita-Latn",
        "nl_nl": "nld-Latn",
        "pl_pl": "pol-Latn",
        "ro_ro": "ron-Latn",
        "ru_ru": "rus-Cyrl",
        "pl_pl": "pol-Latn",
        "te_in": "tel-Telu",
        "hi_in": "hin-Deva",
        "bn_in": "ben-Beng",
        "ta_in": "tam-Taml",
        "ml_in": "mal-Mlym",
        "uk_ua": "ukr-Cyrl",
        "yo_ng": "yor-Latn",
        "xh_za": "xho-Latn",
        "pa_in": "pan-Guru",
        "jv_id": "jav-Latn",
        #"km_kh": "Khmer",
        # "my_mm"
        # "fi_fi"
        # "gl_es"
        # "ja_jp"
        # "gu_in"
        # "sk_sk"
        # "sl_si"
        # "et_ee"
        # "lt_lt"
        # "af_za"
        # "el_gr"
        # "ne_np"
    }


if __name__ == "__main__":
    #tsv2()
    cc100_non()
    #voxp()