import pathlib

def mailabs():
    results_path = pathlib.Path("/home/saeki/workspace/multilingual-tts/NISQA/mailabs_results")
    csvpaths = list(results_path.glob("*/*.csv"))

    wavstem2uttid = {}
    tsv_path = pathlib.Path("/home/saeki/workspace/ssd1/MasMulTTS/m_ailabs.tsv")

    with open(tsv_path, "r") as fr:
        for line in fr:
            line_list = line.strip().split("\t")
            uttid = line_list[0]
            wavpath = pathlib.Path(line_list[1])
            wavstem = wavpath.stem
            wavstem2uttid[wavstem] = uttid

    out_list = ["file_path,mos_pred,noi_pred,dis_pred,col_pred,loud_pred,model"]
    for cp in csvpaths:
        with open(cp, "r") as fr:
            for i, line in enumerate(fr):
                if i == 0:
                    continue
                line_list = line.strip().split(",")
                wavpath = line_list[0]
                wavstem = pathlib.Path(wavpath).stem
                uttid = wavstem2uttid[wavstem]
                scores = line_list[1:]
                new_line = uttid + "," + ",".join(scores)
                out_list.append(new_line)
    
    with open("nisqa_results_mailabs.csv", "w") as fw:
        fw.write("\n".join(out_list))


def fleurs():
    results_path = pathlib.Path("/home/saeki/workspace/multilingual-tts/NISQA/fleurs_results")
    csvpaths = list(results_path.glob("*/*/*.csv"))

    wavstem2uttid = {}
    tsv_path = pathlib.Path("/home/saeki/workspace/ssd1/MasMulTTS/fleurs.tsv")

    with open(tsv_path, "r") as fr:
        for line in fr:
            line_list = line.strip().split("\t")
            uttid = line_list[0]
            wavpath = pathlib.Path(line_list[1])
            wavstem = wavpath.stem
            wavstem2uttid[wavstem] = uttid

    out_list = ["file_path,mos_pred,noi_pred,dis_pred,col_pred,loud_pred,model"]
    for cp in csvpaths:
        with open(cp, "r") as fr:
            for i, line in enumerate(fr):
                if i == 0:
                    continue
                line_list = line.strip().split(",")
                wavname = line_list[0]
                wavstem = wavname.strip().split(".")[0]
                uttid = wavstem2uttid[wavstem]
                scores = line_list[1:]
                new_line = uttid + "," + ",".join(scores)
                out_list.append(new_line)
    
    with open("nisqa_results_fleurs.csv", "w") as fw:
        fw.write("\n".join(out_list))

if __name__ == "__main__":
    fleurs()