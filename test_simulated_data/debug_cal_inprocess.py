import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from datasets.generateData import load_data
from Our_metrics.Scatter_Metrics import Scatter_Metric

file_location = 'datasets/simulated_datasets/csv_files_1/clusternumber2_datanumber500_testnumbercategorymedium_repeatnumber1.csv'

data = load_data(file_location)

analysis = Scatter_Metric(data, 
                        margins={'left': 0.15, 'right': 0.75, 'top': 0.9, 'bottom': 0.1}, 
                        marker='square', 
                        marker_size=60, 
                        dpi=100, 
                        figsize=(10, 8), 
                        xvariable='1', 
                        yvariable='2', 
                        zvariable='class', 
                        color_map='tab10')

analysis.importance_metric(important_cal_method = 'mahalanobis_distance', weight_diff_class=10, weight_same_class=0)

# 1) Call cal_covered_data_points (in-process)
print('--- Running cal_covered_data_points() (in-process) ---')
res = analysis.cal_covered_data_points()
print('cal_covered_data_points returned:', res)

# 2) Analyzer logic using in-memory pixel_color_matrix and covered_pixels from analysis.data
matrix = analysis.pixel_color_matrix
id_to_class = dict(zip(analysis.data['ID'], analysis.data['class']))

# Build pixels_per_id from pixel_color_matrix stacks
from collections import defaultdict
pixels_per_id = defaultdict(set)
any_pixel_top_diff = defaultdict(bool)
all_pixels_top_diff = defaultdict(lambda: True)

H = matrix.shape[0]
W = matrix.shape[1]

for y in range(H):
    for x in range(W):
        stack = matrix[y, x]
        if not stack:
            continue
        top = stack[-1]
        top_cat = top.get('category') if isinstance(top, dict) else None
        top_id = top.get('ID') if isinstance(top, dict) else None
        for elem in stack:
            if not isinstance(elem, dict):
                continue
            eid = elem.get('ID')
            if eid is None:
                continue
            pixels_per_id[eid].add((x,y))
            if top_cat is not None:
                if top_cat != id_to_class.get(eid):
                    any_pixel_top_diff[eid] = True
                else:
                    all_pixels_top_diff[eid] = False
            else:
                all_pixels_top_diff[eid] = False
            if top_id == eid:
                all_pixels_top_diff[eid] = False

ids_with_pixels = [i for i in id_to_class.keys() if i in pixels_per_id and len(pixels_per_id[i])>0]
num_with_any_pixel_covered_by_diff = sum(1 for i in ids_with_pixels if any_pixel_top_diff[i])
num_fully_covered_by_diff = sum(1 for i in ids_with_pixels if all_pixels_top_diff[i])
fully_by_diff_ids = [i for i in ids_with_pixels if all_pixels_top_diff[i]]

print('\nAnalyzer logic (in-memory pixel matrix):')
print('Number with any differing-top pixels:', num_with_any_pixel_covered_by_diff)
print('Number fully covered by diff (analyzer):', num_fully_covered_by_diff)
print('IDs fully covered by diff (analyzer):', fully_by_diff_ids)

# 3) Reconstruct coverage from analysis.data['covered_pixels'] (recomputed by cal_covered_data_points earlier)
covered_from_df = dict()
for _, row in analysis.data.iterrows():
    cp = row['covered_pixels']
    if not isinstance(cp, list):
        covered_from_df[row['ID']] = []
    else:
        covered_from_df[row['ID']] = cp

fully_covered_by_diff_from_df = []
for id_, covered in covered_from_df.items():
    if not covered:
        continue
    is_fully_covered_by_diff = True
    for pix in covered:
        px, py = int(pix[0]), int(pix[1])
        stack = matrix[py, px]
        if not stack:
            is_fully_covered_by_diff = False
            break
        top = stack[-1]
        top_cat = top.get('category') if isinstance(top, dict) else None
        top_id = top.get('ID') if isinstance(top, dict) else None
        if top_id == id_:
            is_fully_covered_by_diff = False
            break
        if top_cat == id_to_class.get(id_):
            is_fully_covered_by_diff = False
            break
    if is_fully_covered_by_diff:
        fully_covered_by_diff_from_df.append(id_)

print('\nFrom analysis.data["covered_pixels"]:')
print('Number fully covered by diff (from df):', len(fully_covered_by_diff_from_df))
print('IDs fully covered by diff (from df):', fully_covered_by_diff_from_df)

# Print intersection/differences
set_cal = set(res if isinstance(res, (list,tuple)) else [])
set_an = set(fully_by_diff_ids)
set_df = set(fully_covered_by_diff_from_df)
print('\nSets: cal_return, analyzer_ids, df_ids')
print('cal_return:', res)
print('analyzer_ids:', sorted(list(set_an)))
print('df_ids:', sorted(list(set_df)))

print('\nDifferences:')
print('in analyzer not in cal_return:', sorted(list(set_an - set_cal)))
print('in cal_return not in analyzer:', sorted(list(set_cal - set_an)))
print('in df not in analyzer:', sorted(list(set_df - set_an)))
