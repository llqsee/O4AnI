from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D

def plot_all_scatterplots(input_dir: str, output_dir: str):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_path.glob('*.csv'))
    if not csv_files:
        print(f"No CSVs found in {input_path!r}")
        return

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, usecols=['1', '2', 'class'])
        except ValueError:
            print(f"Skipping {csv_file.name}: missing one of ['1','2','class']")
            continue

        if df.empty:
            print(f"Skipping {csv_file.name}: file is empty")
            continue

        # --- encode classes as 0,1,2,...
        cat       = pd.Categorical(df['class'])
        codes     = cat.codes
        labels    = cat.categories
        n_classes = len(labels)

        # --- build a discrete colormap of exactly n_classes colors
        base_cmap = plt.get_cmap('tab20').colors  # tuple of 20 base colors
        if n_classes <= len(base_cmap):
            colors = base_cmap[:n_classes]
        else:
            # fallback to evenly sampling a larger cmap
            colors = plt.get_cmap('tab20', n_classes).colors
        cmap = ListedColormap(colors)

        # --- explicit normalization so 0→first color, n-1→last color
        norm = Normalize(vmin=0, vmax=n_classes - 1)

        # --- plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(df['1'], df['2'],
                        c=codes,
                        cmap=cmap,
                        norm=norm,
                        alpha=0.7,
                        edgecolor='k',
                        linewidth=0.2)

        # --- build legend handles that exactly match scatter colors
        handles = []
        for idx, lab in enumerate(labels):
            handles.append(
                Line2D([0], [0],
                       marker='o',
                       color='w',
                       label=str(lab),
                       markerfacecolor=cmap(idx),
                       markersize=8,
                       markeredgecolor='k',
                       markeredgewidth=0.2)
            )
        ax.legend(handles=handles, title='Class', loc='best', frameon=True)

        ax.set_title(f"Scatterplot: {csv_file.stem}")
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')

        out_file = output_path / f"{csv_file.stem}.png"
        fig.savefig(out_file, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"Saved: {out_file}")

if __name__ == "__main__":
    # adjust these paths as needed
    plot_all_scatterplots(
        input_dir  = 'datasets/Sedlmair_data/DR_data',
        output_dir = 'datasets/Sedlmair_data/ExampleFigures'
    )
    print("All scatterplots have been generated and saved.")

    # --- (optional) build a summary CSV of separation labels
    base_out = Path('datasets/Sedlmair_data/ExampleFigures')
    data_dir  = Path('datasets/Sedlmair_data/DR_data')
    summary_file = base_out / 'separation_label.csv'   # fixed typo

    with summary_file.open('w') as f:
        f.write('DatasetName,WellSeparated\n')
        for csv_file in sorted(data_dir.glob('*.csv')):
            f.write(f"{csv_file.name},no\n")  # placeholder logic

    print(f"Summary file created: {summary_file}")
