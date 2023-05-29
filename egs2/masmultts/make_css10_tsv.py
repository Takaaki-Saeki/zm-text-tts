import pathlib
import argparse
import re
import unicodedata

def basic_normalizer(s: str) -> str:
    def remove_symbols(s: str):
        return "".join(
            " " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s)
        )
    s = s.lower()
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
    s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
    s = remove_symbols(s).lower()
    s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--css10_dir",required=True, type=pathlib.Path)
    parser.add_argument("--tsv_path",required=True, type=pathlib.Path)
    parser.add_argument("--apply_normalizer", action="store_true")
    args = parser.parse_args()

    trans_paths = list(args.css10_dir.glob("*/*/transcript.txt"))

    with open(args.tsv_path, "w") as fw:
        pass

    for tp in trans_paths:
        out_list = []
        lang = tp.parent.parent.name
        slang = tp.parent.name
        spkname = "css10_" + tp.parent.name
        with open(tp, "r") as fr:
            for line in fr:
                wavpath = line.strip().split("|")[0]
                wavname = wavpath.strip().split("/")[-1].split(".")[0]
                text = line.strip().split("|")[1]
                if args.apply_normalizer:
                    text = basic_normalizer(text)
                relative_path = f"{lang}/{slang}/{wavpath}"
                uttid = f"{spkname}_{wavname}"
                out_line = [uttid, relative_path, lang, spkname, text]
                out_list.append("\t".join(out_line))
    
        with open(args.tsv_path, "a") as fw:
            fw.write("\n".join(out_list))
            fw.write("\n")    

if __name__ == "__main__":
    main()