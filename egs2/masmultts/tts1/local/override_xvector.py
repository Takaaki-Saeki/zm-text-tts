import pathlib
import argparse
import random
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spk_override", required=True, type=pathlib.Path)
    args = parser.parse_args()

    xv_dir = pathlib.Path(
        "dump/xvector")
    data_dir = pathlib.Path(
        "data")

    if not xv_dir.exists():
        raise ValueError(f"Xvector directory {str(xv_dir)} does not exist")
    if not args.spk_override.exists():
        raise ValueError(f"Override file {str(args.spk_override)} does not exist")

    setnames = ["train", "dev", "test"]

    ov_lang2spk = {}
    with open(args.spk_override, "r") as fr:
        for line in fr:
            line_list = line.strip().split()
            ov_lang2spk[line_list[0]] = line_list[1]
    
    for setname in setnames:
        utt2ark = {}
        spk2ark = {}
        utt2spk = {}
        utt2lang = {}
        spk2lang = {}
        spk2utt = defaultdict(list)

        all_utts = set()
        with open(xv_dir / setname / "xvector.scp", "r") as fr:
            for line in fr:
                line_list = line.strip().split()
                all_utts.add(line_list[0])

        with open(data_dir / setname / "utt2spk", "r") as fr:
            for line in fr:
                line_list = line.strip().split()
                if line_list[0] in all_utts:
                    utt2spk[line_list[0]] = line_list[1]
                    spk2utt[line_list[1]].append(line_list[0])
        with open(data_dir / setname / "utt2lang", "r") as fr:
            for line in fr:
                line_list = line.strip().split()
                if line_list[0] in all_utts:
                    utt2lang[line_list[0]] = line_list[1]
        for spk in spk2utt:
            for utt in spk2utt[spk]:
                if utt in all_utts:
                    spk2lang[spk] = utt2lang[spk2utt[spk][0]]
                    break
        
        # Process xvector.scp
        in_utts = []
        out_list_utt = []
        with open(xv_dir / setname / "xvector.scp", "r") as fr:
            for line in fr:
                line_list = line.strip().split()
                in_utts.append(line_list[0])
                utt2ark[line_list[0]] = line_list[1]
        for utt in in_utts:
            lang = utt2lang[utt]
            if lang in ov_lang2spk:
                new_spk = ov_lang2spk[lang]
                new_utt = random.choice(spk2utt[new_spk])
                new_line = f"{utt} {utt2ark[new_utt]}"
            else:
                new_line = f"{utt} {utt2ark[utt]}"
            out_list_utt.append(new_line)
        
        in_spks = []
        out_list_spk = []
        with open(xv_dir / setname / "spk_xvector.scp", "r") as fr:
            for line in fr:
                line_list = line.strip().split()
                in_spks.append(line_list[0])
                spk2ark[line_list[0]] = line_list[1]
        for spk in in_spks:
            lang = spk2lang[spk]
            if lang in ov_lang2spk:
                new_spk = ov_lang2spk[lang]
                new_line = f"{spk} {spk2ark[new_spk]}"
            else:
                new_line = f"{spk} {spk2ark[spk]}"
            out_list_spk.append(new_line)
        
        with open(xv_dir / setname / "xvector.scp", "w") as fw:
            fw.write("\n".join(out_list_utt))
        with open(xv_dir / setname / "xvector.scp.bak", "w") as fw:
            fw.write("\n".join(out_list_utt))
        with open(xv_dir / setname / "spk_xvector.scp", "w") as fw:
            fw.write("\n".join(out_list_spk))

if __name__ == "__main__":
    main()