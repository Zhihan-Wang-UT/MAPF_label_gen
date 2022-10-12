# %%
from itertools import islice
import subprocess
from os import listdir
from datetime import datetime
from pytz import timezone
import pytz
import sys

# %%
dataset_dir = '../../datasets/'
map_dir = f'{dataset_dir}map/'
scen_even_dir = f'{dataset_dir}scen_even/'
scen_random_dir = f'{dataset_dir}scen_random/'
lns_dir = '../../MAPF_LNS2/'
env = [] # on ubuntu leave empty array [] instaed
runtime = 120
Algos = ["EECBS"]
csv_file_prefix = 'lns_run_result'
max_fail_strike = 5
file_start = 0
file_end = 1650

# %%
# from itertools import islice
def getMapFile(scenfile, is_even_scene = False):
    prefix = scen_even_dir if is_even_scene else scen_random_dir
    with open(prefix + scenfile) as fin:
        for line in islice(fin, 1, 2):
            return line.split()[1]  


# %%
def run_scen(scen_file, is_even, runtime=300, algo='CBS', topk=50, csv_file='test.csv'):
    _mapfile = map_dir + getMapFile(scenfile=scen_file,is_even_scene=is_even)
    _scenfile = (scen_even_dir if is_even else scen_random_dir) + scen_file
    commands = [f'{lns_dir}lns', '-m', f'{_mapfile}', '-a', f'{_scenfile}',
     '-o', csv_file, '-k', f'{topk}', '-t', f'{runtime}', f'--initAlgo={algo}']
    print(" ".join(commands))
    result = subprocess.run(env + commands, capture_output=True)
    return result.stdout.decode("utf-8").replace('\n','')
a = run_scen('Berlin_1_256-even-1.scen', True, 60, topk=10)
b = run_scen('Berlin_1_256-even-1.scen', True, 60, topk=10)
a,b

# %%
file_and_attributes = [(x,True) for x in listdir(scen_even_dir)] + [(x,False) for x in listdir(scen_random_dir)]


# %%
def res_failed(res: str):
    if (res.find("InitLNS") != -1):
        return True
    if (res.find("Failed to find an initial solution") != -1):
        return True
        
    return False

# %%
# from datetime import datetime
# from pytz import timezone
# import pytz
def getDayTime():
    # get timestamp
    day_format = "%Y%m%d"
    time_format = "%H%M%S"
    date = datetime.now(tz=pytz.utc).astimezone(timezone('US/Pacific'))
    day = date.strftime(day_format)
    time = date.strftime(time_format)
    return day, time
    

def run():
    day, time = getDayTime()
    logFile = f'lns_run_log_{day}_{time}.log'
    csvFile = f'lns_run_result_{day}_{time}.csv'
    i = 0

    f = open(logFile, 'w')

    file_and_attributes = [(x,True) for x in listdir(scen_even_dir)] + [(x,False) for x in listdir(scen_random_dir)]

    for algo in Algos:
        for file, is_even in file_and_attributes[file_start:file_end] :
            fail_strike = 0
            skip_scen_all_the_way = False
            file_agents_count = len(open(f'{scen_even_dir if is_even else scen_random_dir}{file}').readlines())-1
            for k in range(10,file_agents_count+1,10):
                res = "Skipped"
                if not skip_scen_all_the_way:
                    res = run_scen(file, is_even, runtime=runtime, algo=algo, topk=k, csv_file=csvFile)
                runinfo = f'{i}, {file}, {k}, {runtime}, {algo}, "{res}"\n'
                
                # write to csv
                f.write(runinfo)
                f.flush()
                i += 1

                # deciding whether to trigger skip_scen_all_the_way or not
                if (not skip_scen_all_the_way):
                    if res_failed(res):
                        fail_strike += 1
                        if fail_strike >= max_fail_strike:
                            skip_scen_all_the_way = True
                    else:
                        fail_strike = 0

    f.close()



# %%
# run()

if __name__ == "__main__":
    # execute only if run as a script
    
    file_start = int(sys.argv[1])
    file_end = int(sys.argv[2])
    print(f'executing file from {file_start} to {file_end}')
    run()