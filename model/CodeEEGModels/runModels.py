import os
import sys

from eegModelUtil import set_parameter
from EEGTCNet import start as EEGTCNet_start
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../EEGNet')))
# from EEGNet import start as EEGNet_start
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'EEGConformer')))
from conformer import start as Conformer_start
def main():
    
    #PARAMETER SWEEP
    lrs = [0.1, 0.01, 0.001, 0.0001]
    batch_sizes = [16, 32, 64, 128]
    
    for i in lrs:
        for j in batch_sizes:
            set_parameter('LR', i)
            set_parameter('BATCH_SIZE', j)
            Conformer_start()


if __name__ == '__main__':
    main()