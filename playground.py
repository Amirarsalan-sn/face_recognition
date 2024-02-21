import numpy as np
import csv

for i in range(16):
    with open(f'D:\\new data set\\{i}_test.csv', 'r', newline='') as csv_file:
        j = 0
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) != 120_001:
                print(f'found an incompatible data in line {j} of {i}_test.csv: {len(row)}')
                print(f'label of that line: {row[-1]}')
            j += 1
