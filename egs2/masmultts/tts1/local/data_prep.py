import pathlib
import argparse
import numpy as np
import os
from collections import defaultdict

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

class DataProcessor:
    def __init__(
        self,
        data_type,
        tsv_path,
        token_type,
        mos_filtering=False,
        lang_set=None,
        byte_len_filtering=False,
        spk_set=None,
        n_train_utt=None,
        override_spk_set=None,
    ):
        self.dst_dir = pathlib.Path("data")
        self.data_type = data_type
        self.tsv_path = tsv_path
        self.token_type = token_type
        self.mos_filtering = mos_filtering
        self.byte_len_filtering = byte_len_filtering
        self.mos_thresh = 2.0
        self.byte_len_thresh = 300
        self.seed = 0

        if lang_set is not None:
            with open(lang_set, "r") as fr:
                self.lang_set = [line.strip() for line in fr]
        else:
            self.lang_set = None
        
        if spk_set is not None:
            with open(spk_set, "r") as fr:
                self.spk_set = [line.strip() for line in fr]
        else:
            self.spk_set = None

        if override_spk_set is not None:
            self.override_spk_set = {}
            with open(override_spk_set, "r") as fr:
                for line in fr:
                    lang, spk = line.strip().split()
                    self.override_spk_set[lang] = spk
        else:
            self.override_spk_set = None
        
        self.n_train_utt = n_train_utt

        if self.data_type == "mailabs":
            self.langtable = langtable_mailabs()
            self.data_name = "m_ailabs"
            self.n_dev = 10
            self.n_test = 100
            self.spks_30min = None
        elif self.data_type == "css10":
            self.langtable = langtable_css10()
            self.data_name = "css10"
            self.n_dev = 10
            self.n_test = 100
            self.mos_filtering = False
            self.spks_30min = None
        
        self.mos_filtered_utt = None
        if self.mos_filtering:
            self.mos_filtered_utt = set()
            mos_path = pathlib.Path(f"local/nisqa_results_{data_type}.csv")
            with open(mos_path, "r") as fr:
                for i, line in enumerate(fr):
                    if i == 0:
                        continue
                    line_list = line.strip().split(",")
                    uttid = line_list[0]
                    mos_val = float(line_list[1])
                    if mos_val > self.mos_thresh:
                        self.byte_len_filtered_utt.add(uttid)
        
        self.byte_len_filtered_utt = None
        if self.byte_len_filtering:
            self.byte_len_filtered_utt = set()
            tsv_path_norm = self.tsv_path.parent / f"{self.data_name}.tsv"
            with open(tsv_path_norm, "r") as fr:
                for line in fr:
                    line_list = line.strip().split("\t")
                    if len(line_list) < 5:
                        continue
                    uttid = line_list[0]
                    text = line_list[4]
                    byte_len = len(list(text.encode("utf-8")))
                    if byte_len <= self.byte_len_thresh:
                        self.byte_len_filtered_utt.add(uttid)

    def get_mos_filtered_uttids(self, utt_list):
        print(f"Filtering utterances with MOS value: {self.mos_thresh}")
        out_utt_list = [
            uttid for uttid in utt_list if uttid in self.mos_filtered_utt]
        return out_utt_list

    def get_byte_len_filtered_uttids(self, utt_list):
        print(f"Filtering utterances with byte lengths: {self.byte_len_thresh}")
        out_utt_list = [
            uttid for uttid in utt_list if uttid in self.byte_len_filtered_utt]
        return out_utt_list

    def remove_non_printable_chars(self, in_string):
        return ''.join(c for c in in_string if c.isprintable())

    def process(self):

        db_dir = self.tsv_path.parent

        utt2spk = {}
        utt2lang = {}
        utt2wav = {}
        utt2text = {}

        self.spk30min_filtered_utt = None
        if self.spks_30min is not None:
            self.spk30min_filtered_utt = set()

        tsvs = [self.tsv_path]
        suffixes = [""]
        lang2utt_list = []

        for tsv_path, suffix in zip(tsvs, suffixes):
            lang2utt = defaultdict(list)
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
                    wavpath = db_dir / self.data_name / line_list[1]
                    lang = line_list[2]
                    spk = line_list[3]
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
                    if self.spk_set is not None:
                        if spk not in self.spk_set:
                            continue
                    if self.override_spk_set is not None:
                        if lang in self.override_spk_set:
                            spk = self.override_spk_set[lang]
                            uttid_org = uttid
                            uttid = f"{spk}_{uttid_org}"
                            if uttid_org in self.byte_len_filtered_utt:
                                self.byte_len_filtered_utt.add(uttid)
                    if self.spks_30min is not None:
                        if spk in self.spks_30min:
                            self.spk30min_filtered_utt.add(uttid)
                    lang2utt[lang].append(uttid)
                    utt2spk[uttid] = spk
                    utt2lang[uttid] = lang
                    utt2wav[uttid] = wavpath
                    utt2text[uttid] = text
            lang2utt_list.append(lang2utt)

        uttids_all = {"train": [], "dev": [], "test": []}

        uttids_test = []
        test_token_idx = 0 # byte

        for l2u_idx, lang2utt in enumerate(lang2utt_list):
            for lang in lang2utt.keys():
                np.random.seed(self.seed)
                rand_idx = np.random.permutation(len(lang2utt[lang]))
                train_idx = rand_idx[self.n_dev+self.n_test :]
                if self.n_train_utt is not None:
                    train_idx = train_idx[: self.n_train_utt]
                uttids_all["train"] += [lang2utt[lang][idx] for idx in train_idx]
                dev_idx = rand_idx[: self.n_dev]
                uttids_all["dev"] += [lang2utt[lang][idx] for idx in dev_idx]
                if l2u_idx == test_token_idx:
                    test_idx = rand_idx[self.n_dev : self.n_dev+self.n_test]
                    uttids_test += [lang2utt[lang][idx] for idx in test_idx]
        
        # Avoiding leak of test utterances
        for uttid_test in uttids_test:
            uttids_all["test"].append(uttid_test)
        
        for setname in ["train", "dev", "test"]:
            utt_list = uttids_all[setname]
            if setname == "train" and self.mos_filtering:
                utt_list = self.get_mos_filtered_uttids(utt_list)
            if setname == "train" and self.byte_len_filtering:
                utt_list = self.get_byte_len_filtered_uttids(utt_list)
            if setname == "train" and self.spks_30min is not None:
                utt_list = self.get_30min_spk_filtered_uttids(utt_list)

            utt2lang_list = []
            wavscp_list = []
            utt2spk_list = []
            text_list = []
            d_spk2utt = defaultdict(list)
            for uttid in utt_list:
                utt2lang_list.append(f"{uttid} {utt2lang[uttid]}")
                wavscp_list.append(f"{uttid} {utt2wav[uttid]}")
                utt2spk_list.append(f"{uttid} {utt2spk[uttid]}")
                text_list.append(f"{uttid} {utt2text[uttid]}")
                d_spk2utt[utt2spk[uttid]].append(uttid)
            spk2utt_list = [f"{spk} {' '.join(utt_list)}" for spk, utt_list in d_spk2utt.items()]

            destination = self.dst_dir / self.data_type / setname
            os.makedirs(destination, exist_ok=True)
            with open(destination / "utt2lang", "w") as fw:
                fw.write("\n".join(utt2lang_list))
            with open(destination / "utt2spk", "w") as fw:
                fw.write("\n".join(utt2spk_list))
            with open(destination / "spk2utt", "w") as fw:
                fw.write("\n".join(spk2utt_list))
            with open(destination / "text", "w") as fw:
                fw.write("\n".join(text_list))
            with open(destination / "wav.scp", "w") as fw:
                fw.write("\n".join(wavscp_list))

def merge_data_set(data_types, setname):
    dst_dir = pathlib.Path("data")
    os.makedirs(dst_dir / setname, exist_ok=True)
    for fname in ["utt2lang", "utt2spk", "spk2utt", "text", "wav.scp"]:
        out_list = []
        for data_type in data_types:
            with open(dst_dir / data_type / setname / fname, "r") as fr:
                out_list += [line.strip() for line in fr]
        with open(dst_dir / setname / fname, "w") as fw:
            fw.write("\n".join(out_list))

def merge_data(data_types):
    for setname in ["train", "dev", "test"]:
        merge_data_set(data_types, setname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir",required=True, type=pathlib.Path)
    parser.add_argument("--token_type", required=True, type=str, choices=["byte", "phn"])
    parser.add_argument("--use_mailabs", action="store_true")
    parser.add_argument("--use_css10", action="store_true")
    parser.add_argument("--mos_filtering", action="store_true")
    parser.add_argument("--byte_len_filtering", action="store_true")
    parser.add_argument("--lang_set", default=None, type=pathlib.Path)
    parser.add_argument("--spk_set", required=False, default=None, type=pathlib.Path)
    parser.add_argument("--override_spk_set", required=False, default=None, type=pathlib.Path)
    parser.add_argument("--n_train_utt", required=False, default=None, type=int)
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
            args.mos_filtering,
            args.lang_set,
            args.byte_len_filtering,
            args.spk_set,
            args.n_train_utt,
            args.override_spk_set).process()
        data_types.append("mailabs")

    if args.use_css10:
        print("Processing CSS10 ...")
        tsv_path = args.db_dir / f"css10{suffix}.tsv"
        DataProcessor(
            "css10",
            tsv_path,
            args.token_type,
            args.mos_filtering,
            args.lang_set,
            args.byte_len_filtering,
            args.spk_set,
            args.n_train_utt,
            args.override_spk_set).process()
        data_types.append("css10")
    
    assert len(data_types) > 0, "No data type is specified."

    print("Merging all the data ...")
    merge_data(data_types)

if __name__ == "__main__":
    main()
