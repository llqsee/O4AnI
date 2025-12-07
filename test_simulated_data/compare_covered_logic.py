import json
import pandas as pd
import os
from collections import defaultdict

json_path = os.path.join('test_simulated_data', 'pixel_color_matrix_category_basedmarkersize60_clusternumber2_datanumber500_testnumbercategorymedium_repeatnumber1.json')
csv_path = os.path.join('test_simulated_data', 'analysis_data_category_basedmarkersize60_clusternumber2_datanumber500_testnumbercategorymedium_repeatnumber1.csv')

with open(json_path, 'r') as f:
    matrix = json.load(f)

df = pd.read_csv(csv_path)

id_to_class = dict(zip(df['ID'], df['class']))

# Reimplement cal_covered_data_points logic exactly (from current Scatter_Metrics.py)
fully_covered_any = []
fully_covered_by_diff = []

H = len(matrix)
W = len(matrix[0]) if H>0 else 0

for _, datum in df.iterrows():
    covered = datum.get('covered_pixels', [])
    if pd.isna(covered) or covered == '[]':
        # pandas will store list as string when reading CSV; try to eval
        try:
            covered = eval(covered)
        except Exception:
            covered = []
    # If covered is a float NaN, handle
    if not isinstance(covered, list):
        covered = []
    if not covered:
        continue
    is_fully_covered_any = True
    is_fully_covered_by_diff_class = True
    for pix in covered:
        if not isinstance(pix, (list, tuple)) or len(pix) < 2:
            is_fully_covered_any = False
            is_fully_covered_by_diff_class = False
            break
        px, py = int(pix[0]), int(pix[1])
        if py < 0 or px < 0 or py >= H or px >= W:
            is_fully_covered_any = False
            is_fully_covered_by_diff_class = False
            break
        stack = matrix[py][px]
        if not stack:
            is_fully_covered_any = False
            is_fully_covered_by_diff_class = False
            break
        top = stack[-1]
        top_id = top.get('ID') if isinstance(top, dict) else None
        top_cat = top.get('category') if isinstance(top, dict) else None
        if top_id == datum['ID']:
            is_fully_covered_any = False
            is_fully_covered_by_diff_class = False
            break
        if top_cat == datum.get('class'):
            is_fully_covered_by_diff_class = False
    if is_fully_covered_any:
        fully_covered_any.append(datum['ID'])
    if is_fully_covered_by_diff_class:
        fully_covered_by_diff.append(datum['ID'])

print('Fully covered by any count (replicated cal logic):', len(fully_covered_any))
print('Fully covered by diff class count (replicated cal logic):', len(fully_covered_by_diff))
print('IDs fully covered by diff class (replicated cal logic):', fully_covered_by_diff)

# Now analyzer logic for comparison (previous script logic)
pixels_per_id = defaultdict(set)
any_pixel_top_diff = defaultdict(bool)
all_pixels_top_diff = defaultdict(lambda: True)

for y in range(H):
    for x in range(W):
        stack = matrix[y][x]
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

print('\nAnalyzer logic:')
print('IDs with pixels count:', len(ids_with_pixels))
print('Number with any differing-top pixels:', num_with_any_pixel_covered_by_diff)
print('Number fully covered by diff (analyzer):', num_fully_covered_by_diff)
print('IDs fully covered by diff (analyzer):', fully_by_diff_ids)

# Show differences
set_cal = set(fully_covered_by_diff)
set_an = set(fully_by_diff_ids)
print('\nIDs in analyzer but not in cal_covered_data logic:', sorted(list(set_an - set_cal)))
print('IDs in cal_covered_data logic but not in analyzer:', sorted(list(set_cal - set_an)))
