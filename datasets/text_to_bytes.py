#!/usr/bin/env python3
import numpy as np

with open('landmarks_in_bytes.txt', 'w', encoding='latin1') as outfile:
    with open('landmarks.txt', 'r') as f:
        print('counting...')
        lines = list(iter(f))
        lines_len = len(lines)
        print(lines_len)
        print('start')
        all_num = []
        for idx, line in enumerate(lines):
            print(f'{(idx + 1)}/{lines_len}')
            nums = map(float, line.strip().split(' '))
            nums = np.array(list(nums), dtype=np.float32)
            all_num.append(nums)
        all_num = np.array(all_num).tobytes().decode('latin1')
        outfile.write(all_num)
