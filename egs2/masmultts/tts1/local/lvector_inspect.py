#!/usr/bin/env python3
#  2022, The University of Tokyo; Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
from tqdm.contrib import tqdm
from sklearn.manifold import TSNE
import lang2vec.lang2vec as l2v
import matplotlib.pyplot as plt
import tqdm

def main():
    langs = [
        "eng",
        "spa",
        "rus",
        "ita",
        "deu",
        "nld",
        "ukr",
        "pol",
        "fin",
        "hun",
        "ell",
        "fra"
    ]
    markers = [".", "o", "v", "^", "<", ">", "+", "x", "X", "D", "d", "|"]
    items = ["phonology_knn", "inventory_knn", "syntax_knn"]
    for item in items:
        vecs = []
        for lang in tqdm.tqdm(langs):
            out = l2v.get_features(lang, item)
            vec = np.asarray(out[lang])
            vecs.append(vec)
        vecs = np.asarray(vecs)
        print(f"{item}: {vecs.shape}")
        v_embed = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=2).fit_transform(vecs)
        print(v_embed.shape)
        plt.figure(figsize=(6, 6))
        for n in range(len(langs)):
            plt.plot(v_embed[n, 0], v_embed[n, 1], label=langs[n], marker=markers[n])
        plt.legend()
        plt.show()
        plt.savefig("{}.png".format(item))
        plt.close()


if __name__ == "__main__":
    main()