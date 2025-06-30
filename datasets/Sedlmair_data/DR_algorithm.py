import pandas as pd
import glob, os
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA

def apply_dr_to_csv(input_path, output_path, method='umap', random_state=42, **kwargs):
    df       = pd.read_csv(input_path)
    features = df.drop(columns=['class'])
    labels   = df['class']

    if method.lower() == 'tsne':
        model = TSNE(n_components=2, random_state=random_state, **kwargs)
    elif method.lower() == 'umap':
        model = UMAP(n_components=2, random_state=random_state, **kwargs)
    elif method.lower() == 'pca':
        model = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    comps = model.fit_transform(features)
    df_dr = pd.DataFrame(comps, columns=['1', '2'])
    df_dr['class'] = labels.values
    df_dr.to_csv(output_path, index=False)

def batch_process_dr(input_dir, output_dir, method='umap', random_state=42, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    for in_file in glob.glob(os.path.join(input_dir, '*.csv')):
        base, _ = os.path.splitext(os.path.basename(in_file))
        out_file = os.path.join(output_dir, f"{base}_{method}.csv")
        apply_dr_to_csv(in_file, out_file, method=method,
                        random_state=random_state, **kwargs)
        print(f"[{method}] {in_file} → {out_file}")

if __name__ == "__main__":
    batch_process_dr(
        input_dir  = "datasets/Sedlmair_data/original_data",
        output_dir = "datasets/Sedlmair_data/pca_data",
        method     = "pca",
        random_state = 42,
        n_neighbors = 15,
        min_dist    = 0.1
    )
