import numpy as np
import subprocess
import re
import os
import sys
import json

MAX_PLACE_HOLDER = 9999
MIN_PLACE_HOLDER = -9999
WORK_DIR = "/project/xli281_uksr/xwu/homoApplication/"
os.chdir(WORK_DIR)
DATA_NAME = "NYX"
d = 3
dims = [512, 512, 512]


def run_test_single(Bsize, eb, state, exe, data_dir, fname_list):
    stat_list, cr_list, elapsed_time_list, error_list = [], [], [], []
    for fname in fname_list:
        path = data_dir + fname
        command = [
            "./build/test/" + exe,
            "./setting/var.{}d.json".format(d),
            path,
            state
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        output = result.stdout
        stat_match = re.search(r"variance = ([+-]?\d*\.\d+)", output)
        cr_match = re.search(r"cr = ([+-]?\d*\.\d+)", output)
        elapsed_time_match = re.search(r"elapsed_time = ([+-]?\d*\.\d+)", output)
        error_match = re.search(r"error = ([+-]?\d*\.\d+)", output)
        if stat_match:
            stat_list.append(float(stat_match.group(1)))
        if cr_match:
            cr_list.append(float(cr_match.group(1)))
        if elapsed_time_match:
            elapsed_time_list.append(float(elapsed_time_match.group(1)))
        if error_match:
            error_list.append(float(error_match.group(1)))

    return np.array(stat_list), np.array(cr_list), np.array(elapsed_time_list), np.array(error_list)


def run_test(B_list, eb_list, state_list, exe):
    stat_dict, cr_dict, time_dict = {}, {}, {}
    for Bsize in B_list:
        for eb in eb_list:
            # write settings
            settings = {
                "dim1": dims[0],
                "dim2": dims[1],
                "dim3": dims[2],
                "B": int(Bsize),
                "eb": float(eb)
            }
            json_path = WORK_DIR + "setting/var.{}d.json".format(d)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
            # run test
            for state in state_list:
                stat_list, cr_list, elapsed_time_list, error_list = run_test_single(Bsize, eb, state, exe, DATA_DIR, fname_list)
                cr_dict[(Bsize, eb)] = 1.0 / np.mean(1.0 / cr_list)
                time_dict[(Bsize, eb, state)] = MAX_PLACE_HOLDER
                stat_dict[(Bsize, eb, state)] = []
                if error_list.size > 0:
                    total_time = elapsed_time_list.sum()
                    time_dict[(Bsize, eb, state)] = total_time
                    stat_dict[(Bsize, eb, state)] = np.array(stat_list)
    return stat_dict, cr_dict, time_dict
            

def relative_error_nonzero(approx, true):
    mask = (true != 0)
    return np.max(np.abs(approx[mask] - true[mask]) / np.abs(true[mask])) if np.any(mask) else None


eb_list = ["1e-1", "1e-2", "1e-3", "1e-4"]
B_list = ["4", "8", "16"]
B_list = ["8"]
state_list = ["0", "1", "2"]
exe_list = ["test_szp_variance_{}d".format(d), "test_szx_variance_{}d".format(d), "test_szr_variance_{}d".format(d)]
DATA_DIR = "/pscratch/xli281_uksr/xwu/datasets/{}/".format(DATA_NAME)
fname_list = os.listdir(DATA_DIR)

# eb_list = ["1e-2"]
# exe_list = ["test_szx_variance_{}d".format(d)]

print("Computing \033[34mVariance\033[0m for {} (dims = {}, {} variables)".format(DATA_NAME, dims[:d], len(fname_list)))

for exe in exe_list:
    stat_dict, cr_dict, time_dict = run_test(B_list, eb_list, state_list, exe)
    for Bsize in B_list:
        for eb in eb_list:
            print("{}, B = {}, eb = {}: aggregated_cr = {}".format(exe, Bsize, eb, cr_dict[(Bsize, eb)]))
            t0, t1, t2 = time_dict[(Bsize, eb, "0")], time_dict[(Bsize, eb, "1")], time_dict[(Bsize, eb, "2")]
            print("state 0: {:.2f}".format(t0))
            print("state 1: {:.2f}".format(t1))
            print("state 2: {:.2f}".format(t2))
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