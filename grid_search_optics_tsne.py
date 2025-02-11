from pathlib import Path

import pandas as pd
import sklearn
from sklearn.manifold import TSNE


def create_folders():
    p: Path = Path("results")
    p_optics = p / "optics"
    p_tsne = p / "tsne"
    p_optics.mkdir(exist_ok=True, parents=True)
    p_tsne.mkdir(exist_ok=True, parents=True)


def run_optics(features):
    p_optics = Path("results/optics/")

    for min_sample in [50, 150, 250, 350, 450]:
        for metric in ["minkowski", "cosine"]:
            optics_run_folder = p_optics / f"run-{min_sample}-{metric}"
            optics_run_folder.mkdir(exist_ok=True, parents=True)
            try:
                clustering = (sklearn.cluster.OPTICS(min_samples=min_sample, metric=metric).fit(features))
                optics_results = pd.DataFrame()
                optics_results["ordering"] = clustering.ordering_
                optics_results["reachability"] = clustering.reachability_
                optics_results["predecessor_"] = clustering.predecessor_
                optics_results["core_distances"] = clustering.core_distances_

                optics_results.to_csv(optics_run_folder / f"optics_result_data.csv", index=False)
            except Exception as e:
                print(e)


def run_tsne(features):
    p_tsne = Path("results/tsne/")

    for perp in [80, 100, 120, 140, 160, 180, 200]:
        for it in [1000, 1500, 2000]:
            for metric in ["cosine", "euclidean"]:
                tsne_result_path = p_tsne / f"tsne-{perp}-{it}-{metric}"
                tsne_result_path.mkdir(exist_ok=True, parents=True)

                try:
                    emb = TSNE(perplexity=perp, max_iter=it, metric=metric, init='random').fit_transform(
                        features)

                    emb_df = pd.DataFrame(emb, columns=["x", "y"])
                    emb_df.to_csv(tsne_result_path / "embeddings.csv", index=False)
                except Exception as e:
                    print(e)


def main():
    data: pd.DataFrame = pd.read_csv("data/node_features.csv")
    features = data[data.columns[2:]]
    create_folders()
    run_optics(features)
    run_tsne(features)


if __name__ == '__main__':
    main()
