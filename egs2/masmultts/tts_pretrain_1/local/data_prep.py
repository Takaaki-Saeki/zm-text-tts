import pathlib
import argparse
import numpy as np
import os
import re
import random
from collections import defaultdict
import tqdm
import unicodedata

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

class DataProcessorVoxp:
    def __init__(
        self,
        db_dir,
        token_type="byte",
        lang_set=None,
        byte_len_filtering=False
    ):
        self.dst_dir = pathlib.Path("data")
        self.token_type = token_type
        if token_type == "byte":
            self.token_suffix=""
        elif token_type == "phn":
            self.token_suffix="_phn"
        self.db_dir = db_dir / "voxp_text" / "lm_data"
        self.data_type = "voxp"
        self.seed = 0

        self.voxp_langs = [
            "en", "de", "es", "et", "cs", "fi", "fr", "hr", "hu", "it", "lt", "nl", "pl", "ro", "sk", "sl"
        ]
        all_langs = [langtable_voxp()[lang] for lang in self.voxp_langs]

        if lang_set is not None:
            with open(lang_set, "r") as fr:
                self.lang_set = [line.strip() for line in fr]
                self.lang_set = [lang for lang in self.lang_set if lang in all_langs]
        else:
            self.lang_set = None

        self.n_dev = 100
        self.n_test = 100

        self.byte_len_filtering = byte_len_filtering
        self.byte_len_thresh = 500
        self.byte_len_filtered_utt = set()

    def get_byte_len_filtered_uttids(self, utt_list):
        print(f"Filtering utterances with byte lengths: {self.byte_len_thresh}")
        out_utt_list = [
            uttid for uttid in utt_list if uttid in self.byte_len_filtered_utt]
        return out_utt_list

    def remove_symbols(self, s: str):
        return "".join(
            " " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s)
        )

    def basic_normalizer(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.remove_symbols(s).lower()
        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s

    def process(self):

        out_db_dir = self.dst_dir / self.data_type
        if out_db_dir.exists():
            print("Skipping data processing as it is already done.")
            return

        for setname in ["train", "dev", "test"]:
            destination = self.dst_dir / self.data_type / setname
            os.makedirs(destination, exist_ok=True)
            with open(destination / "utt2lang", "w") as fw:
                pass
            with open(destination / "text", "w") as fw:
                pass

        for lang in self.voxp_langs:
            utt2text = {}
            langutt = []
            lname = langtable_voxp()[lang]
            if self.lang_set is not None:
                if lname not in self.lang_set:
                    continue
                text_path = self.db_dir / lang / f"sentences{self.token_suffix}.txt"
                with open(text_path, "r") as fr:
                    in_list = [line.strip() for line in fr]
            print(f"Processing {lang} ...")
            cnt_removed = 0
            for idx, text in tqdm.tqdm(enumerate(in_list)):
                index = "0"*(10 - len(str(idx))) + str(idx)
                uttid = f"{self.data_type}_{lname}_{index}"
                if self.token_type == "byte":
                    processed_text = self.basic_normalizer(text)
                else:
                    processed_text = text
                processed_text = processed_text.strip()
                if processed_text == "":
                    cnt_removed += 1
                    continue
                langutt.append(uttid)
                utt2text[uttid] = processed_text
                # Byte length filtering
                if self.token_type == "byte":
                    byte_len = len(list(processed_text.encode("utf-8")))
                else:
                    byte_len = len(processed_text.split())
                if byte_len <= self.byte_len_thresh:
                    self.byte_len_filtered_utt.add(uttid)
            print("Removed {} utterances".format(cnt_removed))

            np.random.seed(self.seed)
            rand_idx = np.random.permutation(len(langutt))
            uttids_all = {}
            train_idx = rand_idx[self.n_dev+self.n_test :]
            uttids_all["train"] = [langutt[idx] for idx in train_idx]
            if self.byte_len_filtering:
                uttids_all["train"] = self.get_byte_len_filtered_uttids(uttids_all["train"])
            dev_idx = rand_idx[: self.n_dev]
            uttids_all["dev"] = [langutt[idx] for idx in dev_idx]
            test_idx = rand_idx[self.n_dev : self.n_dev+self.n_test]
            uttids_all["test"] = [langutt[idx] for idx in test_idx]
            for setname in ["train", "dev", "test"]:
                destination = self.dst_dir / self.data_type / setname
                for uttid in uttids_all[setname]:
                    utt2lang_line = f"{uttid} {lname}"
                    text_line = f"{uttid} {utt2text[uttid]}"
                    with open(destination / "utt2lang", "a") as fw:
                        fw.write(utt2lang_line)
                        fw.write("\n")
                    with open(destination / "text", "a") as fw:
                        fw.write(text_line)
                        fw.write("\n")
            del in_list
            del uttids_all
            del langutt

class DataProcessor:
    def __init__(
        self,
        data_type,
        tsv_path,
        token_type,
        lang_set=None,
        byte_len_filtering=False
    ):
        self.dst_dir = pathlib.Path("data")
        self.data_type = data_type
        self.tsv_path = tsv_path
        self.token_type = token_type
        self.byte_len_filtering = byte_len_filtering
        self.byte_len_thresh = 500
        self.seed = 0

        if lang_set is not None:
            with open(lang_set, "r") as fr:
                self.lang_set = [line.strip() for line in fr]
        else:
            self.lang_set = None

        if self.data_type == "mailabs":
            self.langtable = langtable_mailabs()
            self.data_name = "m_ailabs"
            self.n_dev = 10
            self.n_test = 100
        elif self.data_type == "css10":
            self.langtable = langtable_css10()
            self.data_name = "css10"
            self.n_dev = 10
            self.n_test = 100
        
        self.byte_len_filtered_utt = set()

    def get_byte_len_filtered_uttids(self, utt_list):
        print(f"Filtering utterances with byte lengths: {self.byte_len_thresh}")
        out_utt_list = [
            uttid for uttid in utt_list if uttid in self.byte_len_filtered_utt]
        return out_utt_list

    def remove_non_printable_chars(self, in_string):
        return ''.join(c for c in in_string if c.isprintable())

    def process(self):

        lang2utt = defaultdict(list)
        utt2lang = {}
        utt2text = {}

        tsvs = [self.tsv_path]
        suffixes = [""]

        for tsv_path, suffix in zip(tsvs, suffixes):

            with open(tsv_path, "r") as fr:
                for line in fr:
                    line_list = line.strip().split("\t")
                    if len(line_list) != 5:
                        # Filtering out invalid data
                        continue
                    if len(line_list[1].split(".")) != 2:
                        # Filtering out invalid data
                        continue
                    elif line_list[1].split(".")[-1] != "wav":
                        # Filtering out invalid data
                        continue
                    uttid = line_list[0]+suffix
                    lang = line_list[2]
                    text = line_list[4]
                    if self.token_type == "byte":
                        # Removing invalid characters
                        text = self.remove_non_printable_chars(text)
                        text = text.replace("\u3000", " ")
                        text = text.lower()
                    if self.langtable is not None:
                        lang = self.langtable[lang]
                    if self.lang_set is not None:
                        if lang not in self.lang_set:
                            continue
                    # Byte length filtering
                    if self.token_type == "byte":
                        byte_len = len(list(text.encode("utf-8")))
                    else:
                        byte_len = len(text.split())
                    if byte_len <= self.byte_len_thresh:
                        self.byte_len_filtered_utt.add(uttid)
                    lang2utt[lang].append(uttid)
                    utt2lang[uttid] = lang
                    utt2text[uttid] = text

        uttids_all = {"train": [], "dev": [], "test": []}

        for lang in lang2utt.keys():
            np.random.seed(self.seed)
            rand_idx = np.random.permutation(len(lang2utt[lang]))
            train_idx = rand_idx[self.n_dev+self.n_test :]
            uttids_all["train"] += [lang2utt[lang][idx] for idx in train_idx]
            dev_idx = rand_idx[: self.n_dev]
            uttids_all["dev"] += [lang2utt[lang][idx] for idx in dev_idx]
            test_idx = rand_idx[self.n_dev : self.n_dev+self.n_test]
            uttids_all["test"] += [lang2utt[lang][idx] for idx in test_idx]
        
        for setname in ["train", "dev", "test"]:
            utt_list = uttids_all[setname]
            if setname == "train" and self.byte_len_filtering:
                utt_list = self.get_byte_len_filtered_uttids(utt_list)

            utt2lang_list = []
            text_list = []
            for uttid in utt_list:
                utt2lang_list.append(f"{uttid} {utt2lang[uttid]}")
                text_list.append(f"{uttid} {utt2text[uttid]}")

            destination = self.dst_dir / self.data_type / setname
            os.makedirs(destination, exist_ok=True)
            with open(destination / "utt2lang", "w") as fw:
                fw.write("\n".join(utt2lang_list))
            with open(destination / "text", "w") as fw:
                fw.write("\n".join(text_list))
        del uttids_all

def merge_data_set(data_types, setname):
    dst_dir = pathlib.Path("data")
    os.makedirs(dst_dir / setname, exist_ok=True)
    for fname in ["utt2lang", "text"]:
        # Initialize each file
        with open(dst_dir / setname / fname, "w") as fw:
            pass
        for data_type in data_types:
            print(f"Writing {data_type} ...")
            with open(dst_dir / data_type / setname / fname, "r") as fr:
                for line in tqdm.tqdm(fr):
                    with open(dst_dir / setname / fname, "a") as fw:
                        fw.write(line.strip())
                        fw.write("\n")

def merge_data(data_types):
    for setname in ["train", "dev", "test"]:
        merge_data_set(data_types, setname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir",required=True, type=pathlib.Path)
    parser.add_argument("--token_type", required=True, type=str, choices=["byte", "phn"])
    parser.add_argument("--use_mailabs", action="store_true")
    parser.add_argument("--use_css10", action="store_true")
    parser.add_argument("--use_voxp", action="store_true")
    parser.add_argument("--byte_len_filtering", action="store_true")
    parser.add_argument("--lang_set", default=None, type=pathlib.Path)
    args = parser.parse_args()

    data_types = []

    if args.token_type == "byte":
        suffix = ""
    else:
        suffix = f"_phn"

    if args.use_mailabs:
        print("Processing M-AILABS ...")
        tsv_path = args.db_dir / f"m_ailabs{suffix}.tsv"
        DataProcessor(
            "mailabs",
            tsv_path,
            args.token_type,
            args.lang_set,
            args.byte_len_filtering).process()
        data_types.append("mailabs")

    if args.use_css10:
        print("Processing CSS10 ...")
        tsv_path = args.db_dir / f"css10{suffix}.tsv"
        DataProcessor(
            "css10",
            tsv_path,
            args.token_type,
            args.lang_set,
            args.byte_len_filtering).process()
        data_types.append("css10")
    
    if args.use_voxp:
        print("Processing VoxPopuli ...")
        DataProcessorVoxp(
            args.db_dir,
            args.token_type,
            args.lang_set,
            args.byte_len_filtering).process()
        data_types.append("voxp")
    
    assert len(data_types) > 0, "No data type is specified."

    print("Merging all the data ...")
    merge_data(data_types)

if __name__ == "__main__":
    main()
