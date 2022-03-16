import numpy as np
import os

widths = [8, 16, 32, 64, 128, 256]
seeds = [1, 2, 3, 4, 5]

comd_st = "python UCIExperiments.py --data 'miniboone'"
comd_end = " > results/logs/log_"
# python density_estimation.py --dataset miniboone --flows 5 --layers 0 --hidden_dim 10 --save

for width in widths:
    for seed in seeds:
        comd = comd_st
        comd += (" -hidden_embedding " + str(width) + " " + str(width) + " " + str(width))
        comd += (" -hidden_derivative " + str(width) + " " + str(width) + " " + str(width))
        comd += (" -embedding_size " + str(width) )
        comd += (" --seed " + str(seed))
        comd += comd_end
        comd += ("hdim_" + str(width))
        comd += ("_seed_" + str(seed))
        print(comd)
        os.system(comd)
