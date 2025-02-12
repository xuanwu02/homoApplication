import numpy as np
import subprocess
import re
import os
import sys
import json

MAX_PLACE_HOLDER = 9999
MIN_PLACE_HOLDER = -9999
WORK_DIR = "/project/xli281_uksr/xwu/tripti/"
os.chdir(WORK_DIR)
DATA_NAME = "CESM"
d = 2
dims = [1800, 3600, 1]


def run_test_single(Bsize, eb, exe, data_dir, fname_list):
    stat_list, cr_list, homo_time_list, doc_time_list, diff_list = [], [], [], [], []
    for fname in fname_list:
        path = data_dir + fname
        command = [
            "./build/" + exe,
            path,
            str(dims[0]),
            str(dims[1]),
            Bsize,
            eb
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        output = result.stdout
        stat_match = re.search(r"variance of homomorphic data: ([+-]?\d*\.\d+)", output)
        cr_match = re.search(r"cr = ([+-]?\d*\.\d+)", output)
        homo_time_match = re.search(r"homomorphic varaince elapsed_time:   ([+-]?\d*\.\d+)", output)
        doc_time_match = re.search(r"decompressed variance elapsed_time:   ([+-]?\d*\.\d+)", output)
        diff_match = re.search(r"rel_diff = ([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", output)
        if stat_match:
            stat_list.append(float(stat_match.group(1)))
        if cr_match:
            cr_list.append(float(cr_match.group(1)))
        if homo_time_match and doc_time_match:
            doc_time_list.append(float(doc_time_match.group(1)))
            homo_time_list.append(float(homo_time_match.group(1)))
        if diff_match:
            diff_list.append(float(diff_match.group(1)))

    return np.array(stat_list), np.array(cr_list), np.array(homo_time_list), np.array(doc_time_list), np.array(diff_list)


def run_test(B_list, eb_list, exe):
    stat_dict, cr_dict, time_dict, diff_dict = {}, {}, {}, {}
    for Bsize in B_list:
        for eb in eb_list:
            stat_list, cr_list, homo_time_list, doc_time_list, diff_list = run_test_single(Bsize, eb, exe, DATA_DIR, fname_list)
            cr_dict[(Bsize, eb)] = 1.0 / np.mean(1.0 / cr_list)
            time_dict[(Bsize, eb)] = (MAX_PLACE_HOLDER, MAX_PLACE_HOLDER)
            stat_dict[(Bsize, eb)] = []
            if diff_list.size > 0:
                doc_total_time = doc_time_list.sum()
                homo_total_time = homo_time_list.sum()
                time_dict[(Bsize, eb)] = (doc_total_time, homo_total_time)
                stat_dict[(Bsize, eb)] = stat_list
                diff_dict[(Bsize, eb)] = np.max(diff_list)
    return stat_dict, cr_dict, time_dict, diff_dict
            

def relative_error_nonzero(approx, true):
    mask = (true != 0)
    return np.max(np.abs(approx[mask] - true[mask]) / np.abs(true[mask])) if np.any(mask) else None


eb_list = ["1e-1", "1e-2", "1e-3", "1e-4"]
B_list = ["4", "8", "16"]
B_list = ["8"]
exe_list = ["example_ompSZp_2D_variance"]
DATA_DIR = "/pscratch/xli281_uksr/xwu/datasets/{}/".format(DATA_NAME)
fname_list = os.listdir(DATA_DIR)

eb_list = ["1e-4"]

print("Computing \033[34mVariance\033[0m for {} (dims = {}, {} variables)".format(DATA_NAME, dims[:d], len(fname_list)))

for exe in exe_list:
    stat_dict, cr_dict, time_dict, diff_dict = run_test(B_list, eb_list, exe)
    for Bsize in B_list:
        for eb in eb_list:
            print("{}, B = {}, eb = {}: aggregated_cr = {}".format(exe, Bsize, eb, cr_dict[(Bsize, eb)]))
            t0, t1 = time_dict[(Bsize, eb)][0], time_dict[(Bsize, eb)][1]
            print("  doc time = {:.6f}".format(t0))
            print("  homo time = {:.6f}".format(t1))
            print("  execution time reduction = {:.2f}%".format((t0 - t1) / t0 * 100))
            print("  max relative difference = {:.2e}".format(diff_dict[(Bsize, eb)]))
            # s0, s1, s2 = stat_dict[(Bsize, eb, "0")], stat_dict[(Bsize, eb, "1")], stat_dict[(Bsize, eb, "2")]
            # if t1 != MAX_PLACE_HOLDER:
            #     print("  state 1 (prePred):")
            #     print("    execution time reduction = \033[32m{:.2f}%\033[0m".format((t0 - t1) / t0 * 100))
            #     print("    max relative difference = \033[31m{:.2e}\033[0m".format( relative_error_nonzero(s1, s0) ))
            # if t2 != MAX_PLACE_HOLDER:
            #     print("  state 2 (postPred):")
            #     print("    execution time reduction = \033[32m{:.2f}%\033[0m".format((t0 - t2) / t0 * 100))
            #     print("    max relative difference = \033[31m{:.2e}\033[0m".format( relative_error_nonzero(s2, s0) ))
            sys.stdout.flush()
    print()