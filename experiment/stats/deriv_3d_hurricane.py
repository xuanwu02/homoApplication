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
DATA_NAME = "Hurricane"
d = 3
dims = [100, 500, 500]


def run_test_single(Bsize, eb, state, exe, data_dir, fname_list):
    cr_list, elapsed_time_list, dx_error_list, dy_error_list, dz_error_list = [], [], [], [], []
    for fname in fname_list:
        path = data_dir + fname
        command = [
            "./build/test/" + exe,
            "./setting/deriv.{}d.json".format(d),
            path,
            state
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        
        output = result.stdout
        cr_match = re.search(r"cr = ([+-]?\d*\.\d+)", output)
        elapsed_time_match = re.search(r"elapsed_time = ([+-]?\d*\.\d+)", output)
        dx_error_match = re.search(r"dx max error = ([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", output)
        dy_error_match = re.search(r"dy max error = ([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", output)
        dz_error_match = re.search(r"dz max error = ([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", output)
        if cr_match:
            cr_list.append(float(cr_match.group(1)))
        if elapsed_time_match:
            elapsed_time_list.append(float(elapsed_time_match.group(1)))
        if dx_error_match:
            dx_error_list.append(float(dx_error_match.group(1)))
        if dy_error_match:
            dy_error_list.append(float(dy_error_match.group(1)))
        if dz_error_match:
            dz_error_list.append(float(dz_error_match.group(1)))
            
    return np.array(cr_list), np.array(elapsed_time_list), np.array(dx_error_list), np.array(dy_error_list), np.array(dz_error_list)


def run_test(B_list, eb_list, state_list, exe):
    cr_dict, time_dict, maxerr_dict = {}, {}, {}
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
            json_path = WORK_DIR + "setting/deriv.{}d.json".format(d)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
            # run test
            for state in state_list:
                cr_list, elapsed_time_list, dx_error_list, dy_error_list, dz_error_list = run_test_single(Bsize, eb, state, exe, DATA_DIR, fname_list)
                cr_dict[(Bsize, eb)] = cr_list[0]
                time_dict[(Bsize, eb, state)] = MAX_PLACE_HOLDER
                maxerr_dict[(Bsize, eb, state)] = (MIN_PLACE_HOLDER, MIN_PLACE_HOLDER, MIN_PLACE_HOLDER)
                if dx_error_list.size > 0:
                    total_time = elapsed_time_list.sum()
                    dx_max_err, dy_max_err, dz_max_err = np.max(dx_error_list), np.max(dy_error_list), np.max(dz_error_list)
                    time_dict[(Bsize, eb, state)] = total_time
                    maxerr_dict[(Bsize, eb, state)] = (dx_max_err, dy_max_err, dz_max_err)
    return cr_dict, time_dict, maxerr_dict
            

# 2D CESM 1800 x 3600
eb_list = ["1e-1", "1e-2", "1e-3", "1e-4"]
B_list = ["4", "8", "16"]
B_list = ["8"]
state_list = ["0", "1", "2"]
exe_list = ["test_szp_derivative_{}d".format(d), "test_szx_derivative_{}d".format(d), "test_szr_derivative_{}d".format(d)]
DATA_DIR = "/pscratch/xli281_uksr/xwu/datasets/{}/".format(DATA_NAME)
fname_list = os.listdir(DATA_DIR)

# eb_list = ["1e-1"]
# B_list = ["8"]
# exe_list = ["test_szp_derivative_{}d".format(d)]
 
print("Computing \033[34mCentral Difference\033[0m for {} (dims = {}, {} variables)".format(DATA_NAME, dims, len(fname_list)))

for exe in exe_list:
    cr_dict, time_dict, maxerr_dict = run_test(B_list, eb_list, state_list, exe)
    for Bsize in B_list:
        for eb in eb_list:
            print("{}, B = {}, eb = {}: aggregated_cr = {}".format(exe, Bsize, eb, cr_dict[(Bsize, eb)]))
            t0, t1, t2 = time_dict[(Bsize, eb, "0")], time_dict[(Bsize, eb, "1")], time_dict[(Bsize, eb, "2")]
            if t1 != MAX_PLACE_HOLDER:
                print("  state 1 (prePred):")
                print("    execution time reduction = \033[32m{:.2f}%\033[0m".format((t0 - t1) / t0 * 100))
                print("    max dx error = \033[31m{:.2e}\033[0m".format(maxerr_dict[(Bsize, eb, "1")][0]))
                print("    max dy error = \033[31m{:.2e}\033[0m".format(maxerr_dict[(Bsize, eb, "1")][1]))
                print("    max dz error = \033[31m{:.2e}\033[0m".format(maxerr_dict[(Bsize, eb, "1")][2]))
            if t2 != MAX_PLACE_HOLDER:
                print("  state 2 (postPred):")
                print("    execution time reduction = \033[32m{:.2f}%\033[0m".format((t0 - t2) / t0 * 100))
                print("    max dx error = \033[31m{:.2e}\033[0m".format(maxerr_dict[(Bsize, eb, "2")][0]))
                print("    max dy error = \033[31m{:.2e}\033[0m".format(maxerr_dict[(Bsize, eb, "2")][1]))
                print("    max dz error = \033[31m{:.2e}\033[0m".format(maxerr_dict[(Bsize, eb, "2")][2]))
            sys.stdout.flush()
    print()