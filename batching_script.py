import os

dataset = ['moons', 'circles', 'rings']
depths = [7, 8]
hiddenunits = [32, 64]
epochs = [10**4]
batchsizes = [64, 256]


for NITERS in epochs:
    for SIZE in batchsizes:
        for DEPTH in depths:
            for UNIT in hiddenunits:
                for DATA in dataset:
                    os.system(f"python3 sgm_ism.py --data {DATA} --depth {DEPTH} --hiddenunits {UNIT} --niters {NITERS} --batch_size {SIZE}")