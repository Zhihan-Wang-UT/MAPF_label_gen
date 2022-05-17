# %%
from itertools import islice
import subprocess
from os import listdir
import os
from datetime import datetime
from pytz import timezone
import pytz
import sys

import pandas as pd
from sympy import is_convex


def getDayTime():
    # get timestamp
    day_format = "%Y%m%d"
    time_format = "%H%M%S"
    date = datetime.now(tz=pytz.utc).astimezone(timezone('US/Pacific'))
    day = date.strftime(day_format)
    time = date.strftime(time_format)
    return day, time

# %%
getDayTime()[0]

# %%
tensorgen_dir = '../build/'
tensorgen_executable = './datagen'
scen_even_dir = '../../datasets/scen_even/'
scen_random_dir = '../../datasets/scen_random/'
map_dir = '../../datasets/map/'
output_dir = f'../tensorgen_out/tensors_{getDayTime()[0]}/'
env = [] # on ubuntu leave empty array [] instaed

# %%
def create_dir_if_not_exist(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

def run_tensorgen_cpp_v3(scen_file: str, is_even: bool, out_tensor_prefix: str, num_agents_cap: int):
    create_dir_if_not_exist(f'{output_dir}/{out_tensor_prefix}/')

    _scenfile = (scen_even_dir if is_even else scen_random_dir) + scen_file
    commands = [f'{tensorgen_dir}{tensorgen_executable}', f'{_scenfile}', f'{map_dir}',
    f'{output_dir}/{out_tensor_prefix}/', f'{out_tensor_prefix}', f'{num_agents_cap}']
    # print(" ".join(commands))
    result = subprocess.run(env + commands, capture_output=True)
    return result

# res = run_tensorgen_cpp_v3('Berlin_1_256-even-1.scen', 1, 'brc_10',30)
# res


# %%
# res1 = res
# res1.stdout
# res1.stderr
# " ".join(res1.args)


    

def test_even(scenfile: str):
    if scenfile.find('.scen') == -1:
        print(f"test_even(scenfile) must takes in a scenfile, got {scenfile}")
        return None
    if scenfile.find('even') != -1:
        return True
    return False

def run(file_start = 0, file_end = 1650):
    create_dir_if_not_exist(output_dir)
    
    day, time = getDayTime()
    logFile = f'tensorgen_run_log_{day}_{time}.log'
    i = 0

    f = open(logFile, 'w')

    file_and_num_agent_cap_df = pd.read_csv('scen2num_agent_df_0331.csv')

    print(f"out of {len(file_and_num_agent_cap_df)} files")

    # file_and_attributes = [(x,True) for x in listdir(scen_even_dir)] + [(x,False) for x in listdir(scen_random_dir)]
    

    for _, row in file_and_num_agent_cap_df.iloc[file_start:file_end].iterrows():
        file = row["scenfile"]
        num_agents_cap = row["max_agent_one_solvable"]
        is_even = test_even(file)
        out_tensor_prefix = f'{file.replace(".scen","")}'
        res = run_tensorgen_cpp_v3(file, is_even, out_tensor_prefix, num_agents_cap)
        res_stdout = res.stdout.decode("utf-8").replace('\n','')
        res_info = f'"{" ".join(res.args)}"'
        runinfo = f'{i}, {file}, {out_tensor_prefix}, {res_info}, "{res_stdout}"\n'
        f.write(runinfo)
        f.flush()
        i += 1
    
# run()

# %%
if __name__ == "__main__":
    # execute only if run as a script
    
    file_start = int(sys.argv[1])
    file_end = int(sys.argv[2])
    print(f'executing tensorgen on file from {file_start} to {file_end}')
    run(file_start, file_end)



# %%


