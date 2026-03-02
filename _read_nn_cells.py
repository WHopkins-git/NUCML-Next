import json

with open('notebooks/00_Baselines_and_Limitations.ipynb') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    keywords = ['SimpleNet', 'NN_HIDDEN', 'NN_EPOCHS', 'NN_BATCH',
                'NN_LOSS', 'NN_LEARN', 'NN_TEST', 'nn_model',
                'Baseline 3', 'neural network', 'Neural Network']
    if any(k in src for k in keywords):
        print(f'--- Cell {i} ({cell["cell_type"]}) ---')
        for line in cell['source']:
            print(line, end='')
        print('\n')
