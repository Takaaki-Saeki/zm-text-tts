"""
Multilingual text normalizer.
https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py
"""

import re
import unicodedata
import argparse
import pathlib
import tqdm
import regex

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}

def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        c
        if c in keep
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s)
    )


class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_tsv", default=None, type=pathlib.Path)
    parser.add_argument("--out_tsv", default=None, type=pathlib.Path)
    parser.add_argument("--remove_diacritics", action="store_true")
    parser.add_argument("--split_letters", action="store_true")
    args = parser.parse_args()

    normalizer = BasicTextNormalizer(
        remove_diacritics=args.remove_diacritics, split_letters=args.split_letters)

    out_list = []
    with open(args.in_tsv, "r") as fr:
        in_list = [line.strip() for line in fr]
    for line in tqdm.tqdm(in_list):
        line_list = line.strip().split("\t")
        if len(line_list) < 5:
            continue
        uttid = line_list[0]
        wavpath = line_list[1]
        lang = line_list[2]
        spk = line_list[3]
        text = line_list[4]
        normalized_text = normalizer(text)
        out_line = "\t".join([uttid, wavpath, lang, spk, normalized_text])
        out_list.append(out_line)
    
    with open(args.out_tsv, "w") as fw:
        fw.write("\n".join(out_list))


if __name__ == "__main__":
    main()