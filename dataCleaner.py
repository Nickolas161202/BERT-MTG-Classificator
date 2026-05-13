import json
from collections import defaultdict

def limpar_dataset_magic(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
