import sys, re
import numpy as np

with open(sys.argv[1], 'r') as fhand:
    n_shot = '?'
    n_aug = '?'
    loss_list = []
    acc_list = []
    acc_novel_only_list = []
    acc_novel_list = []
    lr_base = '?'
    lr_power = '?'
    lr = lr_base+'e-'+lr_power
    
    print('lr,n_shot,n_aug,test_loss,novel_only_mean,novel_only_std,novel_mean,novel_std,all_mean,all_std')
    
    str_to_print_all = []
    loss_list_ave_all = []
    acc_novel_list_ave_all = []
    for line in fhand:
        line = line.strip()
        if re.search('^WARNING', line):
            if re.search('_0" already exists!', line):
                ##### if 'acc_list' is not empty, it means that we already collect some results
                ##### ==> print the results and clear 'acc_list' for the next result
                if not len(acc_list) == 0:
                    str_to_print_all.append('%s,%s,%s,%.6f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % \
                        (lr, n_shot, n_aug, np.mean(loss_list), np.mean(acc_novel_only_list)*100, np.std(acc_novel_only_list)*100, np.mean(acc_novel_list)*100, np.std(acc_novel_list)*100, np.mean(acc_list)*100, np.std(acc_list)*100))
                    loss_list_ave_all.append(np.mean(loss_list))
                    acc_novel_list_ave_all.append(np.mean(acc_novel_list))
                    print(str_to_print_all[-1])
                    # print('%s,%s,%s,%.6f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % \
                    #     (lr, n_shot, n_aug, np.mean(loss_list), np.mean(acc_novel_only_list)*100, np.std(acc_novel_only_list)*100, np.mean(acc_novel_list)*100, np.std(acc_novel_list)*100, np.mean(acc_list)*100, np.std(acc_list)*100))
                    loss_list = []
                    acc_list = []
                    acc_novel_only_list = []
                    acc_novel_list = []
                lr_base = re.search('_lr([0-9]+)e[0-9]+_ite', line).group(1)
                lr_power = re.search('_lr[0-9]+e([0-9]+)_ite', line).group(1)
                lr = lr_base+'e-'+lr_power
            ## Extract n_shot and n_aug
            if re.search('shot[0-9]+aug[0-9]+', line):
                n_shot = re.search('shot([0-9]+)aug[0-9]+', line).group(1)
                n_aug = re.search('shot[0-9]+aug([0-9]+)', line).group(1)
        else:
            loss_list.append(float(re.search('test loss: ([0-9]+\.[0-9]+)', line).group(1)))
            acc_list.append(float(re.search('top-5 test accuracy: ([0|1]\.[0-9]+)', line).group(1)))
            acc_novel_only_list.append(float(re.search('novel-only top-5 test accuracy: ([0|1]\.[0-9]+)', line).group(1)))
            acc_novel_list.append(float(re.search('novel top-5 test accuracy: ([0|1]\.[0-9]+)', line).group(1)))
    str_to_print_all.append('%s,%s,%s,%.6f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % \
        (lr, n_shot, n_aug, np.mean(loss_list), np.mean(acc_novel_only_list)*100, np.std(acc_novel_only_list)*100, np.mean(acc_novel_list)*100, np.std(acc_novel_list)*100, np.mean(acc_list)*100, np.std(acc_list)*100))
    loss_list_ave_all.append(np.mean(loss_list))
    acc_novel_list_ave_all.append(np.mean(acc_novel_list))
    print(str_to_print_all[-1])
    # print('%s,%s,%s,%.6f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f' % \
    #     (lr, n_shot, n_aug, np.mean(loss_list), np.mean(acc_novel_only_list)*100, np.std(acc_novel_only_list)*100, np.mean(acc_novel_list)*100, np.std(acc_novel_list)*100, np.mean(acc_list)*100, np.std(acc_list)*100))

# print the best hyper-parameter (with the corresponding l2_reg_power)
print('----------')
# (1) based on the min loss
best_idx = np.argmin(loss_list_ave_all)
# print('best_idx:', best_idx)
print('best test loss: ', end='')
print(str_to_print_all[best_idx])
# (2) based on the max novel top-5 accuracy
best_idx = np.argmax(acc_novel_list_ave_all)
# print('best_idx:', best_idx)
print('best novel top-5 acc: ', end='')
print(str_to_print_all[best_idx])