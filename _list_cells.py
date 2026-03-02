import json

with open('notebooks/00_Baselines_and_Limitations.ipynb') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')
print()
for i, c in enumerate(nb['cells']):
    src = c['source'][0][:80] if c['source'] else '(empty)'
    print(f'Cell {i} ({c["cell_type"]}): {repr(src)}')
