import numpy as np
import pathlib


def main():

    for setname in ["train", "dev", "test"]:
        out_lines = []

        pathname = f"{setname}_text"
        path = pathlib.Path(pathname)
        with open(path, "r") as fr:
            in_lines = fr.readlines()
        
        for line in in_lines:
            out_lines.append(line.strip().split()[0])
        
        with open(pathlib.Path("uttids_paper") / f"{setname}.txt", "w") as fw:
            fw.write("\n".join(out_lines))


if __name__ == "__main__":
    main()