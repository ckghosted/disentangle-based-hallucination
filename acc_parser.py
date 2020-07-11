import sys, re
import numpy as np

with open(sys.argv[1], 'r') as fhand:
    n_shot = '?'
    n_aug = '?'
    acc_list = []
    acc_novel_list = []
    lr_base = '?'
    lr_power = '?'
    lr = lr_base+'e-'+lr_power
    
    print('lr,n_shot,n_aug,novel_mean,novel_std,all_mean,all_std')
    
    for line in fhand:
        line = line.strip()
        if re.search('^WARNING', line):
            if re.search('_0" already exists!', line):
                ##### if 'acc_list' is not empty, it means that we already collect some results
                ##### ==> print the results and clear 'acc_list' for the next result
                if not len(acc_list) == 0:
                    print('%s,%s,%s,%.2f,%.2f,%.2f,%.2f' % \
                        (lr, n_shot, n_aug, np.mean(acc_novel_list)*100, np.std(acc_novel_list)*100, np.mean(acc_list)*100, np.std(acc_list)*100))
                    acc_list = []
                    acc_novel_list = []
                lr_base = re.search('_lr([0-9]+)e[0-9]+_ite', line).group(1)
                lr_power = re.search('_lr[0-9]+e([0-9]+)_ite', line).group(1)
                lr = lr_base+'e-'+lr_power
            ## Extract n_shot and n_aug
            if re.search('shot[0-9]+aug[0-9]+', line):
                n_shot = re.search('shot([0-9]+)aug[0-9]+', line).group(1)
                n_aug = re.search('shot[0-9]+aug([0-9]+)', line).group(1)
        else:
            acc_list.append(float(re.search('top-5 test accuracy: ([0|1]\.[0-9]+)', line).group(1)))
            acc_novel_list.append(float(re.search('novel top-5 test accuracy: ([0|1]\.[0-9]+)', line).group(1)))
    print('%s,%s,%s,%.2f,%.2f,%.2f,%.2f' % \
        (lr, n_shot, n_aug, np.mean(acc_novel_list)*100, np.std(acc_novel_list)*100, np.mean(acc_list)*100, np.std(acc_list)*100))


