import pathlib
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_set", required=True, type=pathlib.Path)
    parser.add_argument("--holdout_lang", required=True, type=pathlib.Path)
    parser.add_argument("--org_dir", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    args = parser.parse_args()

    with open(args.lang_set, "r") as fr:
        langs = [line.strip() for line in fr]
    with open(args.holdout_lang, "r") as fr:
        ho_langs = [line.strip() for line in fr]
    with open(args.config, "r") as fr:
        config = yaml.safe_load(fr)

    # Validating information
    lang2lid = {}
    with open(args.org_dir / "train" / "lang2lid", "r") as fr:
        for line in fr:
            line_list = line.strip().split()
            lang2lid[line_list[0]] = line_list[1]
    for hol in ho_langs:
        assert hol in langs, f"{hol} is not in lang_set."

    # Writing holdout information
    holdout_lids = [lang2lid[lang] for lang in ho_langs]
    config["tts_conf"]["holdout_lids"] = " ".join(holdout_lids)
    out_path = args.config.parent / (args.config.stem+"_override.yaml")
    with open(out_path, "w") as fw:
        yaml.safe_dump(config, fw)
    print(f"Successfully wrote holdout information to {out_path} !")

if __name__ == "__main__":
    main()