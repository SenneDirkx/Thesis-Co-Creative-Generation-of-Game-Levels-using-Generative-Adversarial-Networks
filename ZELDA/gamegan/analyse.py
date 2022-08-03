import os
from math import sqrt
# assign directory
#directory = './copypaste/eval_output/setup2/'
directory = './eval_output_custom_local/setup4/'
CP = False
IGNORE_LOW_ITER = True
# iterate over files in
# that directory

data = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        ftmp = f.split("/")[-1].split('.')
        if ftmp[-1] != 'txt':
            continue
        fd = ".".join(ftmp[:-1]).split("_")[2:]
        #print(fd)
        if CP:
            assert len(fd) == 5 
        else:
            assert len(fd) == 6
        fd1 = list(map(lambda x: int(x[1:]), fd[:-1]))
        fd2 = float(fd[-1][2:])
        fd = fd1 + [fd2]

        if IGNORE_LOW_ITER:
            if CP:
                if fd[1] <= 1000:
                    continue
            else:
                if fd[1] <= 1000 or fd[2] <= 1000:
                    continue

        with open(f, 'r') as datafile:
            lines = datafile.readlines()[12:]
            measurements = list(map(lambda x: float(x.split(" ")[2]), lines))
            fd += measurements
        
        data.append(fd)

print("length data:", len(data))
print("length element:", len(data[0]))

# c, n, l, d, lr,      DSC, LPIPS, PLAYA
# c, v, n, l, d, lr    DSC, LPIPS, PLAYA

def split_on_config(split_category_id, x_category_id, y_category_id, data, col_category_id=None):
    store = {"x": [], "y": [], "split":[], "col": []}

    for elem in data:
        split_val = elem[split_category_id]
        x_val = elem[x_category_id]
        y_val = elem[y_category_id]
        store["split"].append(split_val)
        store["x"].append(x_val)
        store["y"].append(y_val)

        if col_category_id is not None:
             store["col"].append(elem[col_category_id])
    
    return store

def avgs_on_config(split_category_id, output_category_id, data):
    store = {}

    for elem in data:
        split_val = elem[split_category_id]
        output_val = elem[output_category_id]
        if split_val not in store:
            store[split_val] = []
        store[split_val].append(output_val)
    
    for config in store:
        mean = sum(store[config])/len(store[config])
        var_list = [elem - mean for elem in store[config]]
        var = sum([pow(elem, 2) for elem in var_list]) / (len(store[config]) - 1)
        sd = sqrt(var)
        print("Config", config, ": mean of", mean, "with SD of", sd)

def make_avg_sd_table(split_category_id, data):
    store = {}

    if CP and split_category_id > 0:
        split_category_id -= 1

    for elem in data:
        split_val = elem[split_category_id]
        if CP:
            begin_metric = 5
            end_metric = 8
        else:
            begin_metric = 6
            end_metric = 9
        output_val = elem[begin_metric:end_metric]
        if split_val not in store:
            store[split_val] = []
        store[split_val].append(output_val)
    
    sorted_configs = sorted(list(store.keys()))

    result = {}

    for config in sorted_configs:
        result[config] = []
        for metric in range(3):
            metric_data = list(map(lambda x: x[metric], store[config]))
            mean = sum(metric_data)/len(metric_data)
            var_list = [elem - mean for elem in metric_data]
            var = sum([pow(elem, 2) for elem in var_list]) / (len(metric_data) - 1)
            sd = sqrt(var)
            if metric == 2:
                mean *= 100
                sd *= 100
                mean = round(mean, 1)
                sd = round(sd, 1)
            else:
                mean = round(mean, 3)
                sd = round(sd, 3)
            result[config] += [mean, sd]
    

    template_fill = "& {0} & {1} & {2} & {3} & {4} & {5}\\% & {6}\\% \\\\\n"
    

    template = ""
    for config in sorted_configs:
        template += template_fill.format(config, *result[config])
    print(template)

# 5, 0, 3
#make_avg_sd_table(3, data)
#avgs_on_config(0, 7, data)

# LR splitted on LPIPS and PLAYA for both layers
# split, x, y, col = 5, 7, 8, 3

# N iter splitted on LPIPS and PLAYA for both layers
# split, x, y, col = 2, 7, 8, 3

# V iter splitted on LPIPS and PLAYA for both layers
#split, x, y, col = 1, 7, 8, 3
#dict_with_data = split_on_config(split, x, y, data, col)


def count_percentage_good(split_category_id, data):

    BASELINE_DSC = 0.863
    BASELINE_LPIPS = 0.167
    BASELINE_PLAYA = 0.93

    if CP:
        if split_category_id > 0:
            split_category_id -= 1
        begin_metric = 5
        end_metric = 8
    else:
        begin_metric = 6
        end_metric = 9
    
    counter_DSC = {}
    counter_LPIPS = {}
    counter_PLAYA = {}
    counter_all = {}
    
    counter_almost_DSC = {}
    counter_almost_LPIPS = {}
    counter_almost_PLAYA = {}
    counter_almost_all = {}
    totals = {}

    for elem in data:
        split_val = elem[split_category_id]
        output_val = elem[begin_metric:end_metric]
        if split_val not in counter_DSC:
            counter_DSC[split_val] = 0
            counter_LPIPS[split_val] = 0
            counter_PLAYA[split_val] = 0
            counter_all[split_val] = 0
            counter_almost_DSC[split_val] = 0
            counter_almost_LPIPS[split_val] = 0
            counter_almost_PLAYA[split_val] = 0
            counter_almost_all[split_val] = 0
            totals[split_val] = 0
        if output_val[0] <= BASELINE_DSC:
            counter_DSC[split_val] += 1
        if output_val[1] >= BASELINE_LPIPS:
            counter_LPIPS[split_val] += 1
        if output_val[2] >= BASELINE_PLAYA:
            counter_PLAYA[split_val] += 1
        if output_val[0] <= BASELINE_DSC * 1.1:
            counter_almost_DSC[split_val] += 1
        if output_val[1] >= BASELINE_LPIPS * 0.9:
            counter_almost_LPIPS[split_val] += 1
        if output_val[2] >= BASELINE_PLAYA * 0.9:
            counter_almost_PLAYA[split_val] += 1
        
        if output_val[0] <= BASELINE_DSC and output_val[1] >= BASELINE_LPIPS and output_val[2] >= BASELINE_PLAYA:
            counter_all[split_val] += 1
        if output_val[0] <= BASELINE_DSC * 1.1 and output_val[1] >= BASELINE_LPIPS * 0.9 and output_val[2] >= BASELINE_PLAYA * 0.9:
            counter_almost_all[split_val] += 1
        totals[split_val] += 1
    
    sorted_configs = sorted(list(counter_DSC.keys()))

    result = {}

    for config in sorted_configs:
        result[config] = []
        metric_data_dsq = counter_DSC[config]
        metric_data_lpips = counter_LPIPS[config]
        metric_data_playa = counter_PLAYA[config]

        metric_data_dsq_almost = counter_almost_DSC[config]
        metric_data_lpips_almost = counter_almost_LPIPS[config]
        metric_data_playa_almost = counter_almost_PLAYA[config]

        metric_data_all = counter_all[config]
        metric_data_all_almost = counter_almost_all[config]

        total = totals[config]
        percentage_dsq = round(metric_data_dsq/total*100,1)
        percentage_lpips = round(metric_data_lpips/total*100,1)
        percentage_playa = round(metric_data_playa/total*100,1)

        percentage_dsq_almost = round(metric_data_dsq_almost/total*100,1)
        percentage_lpips_almost = round(metric_data_lpips_almost/total*100,1)
        percentage_playa_almost = round(metric_data_playa_almost/total*100,1)

        percentage_all = round(metric_data_all/total*100,1)
        percentage_all_almost = round(metric_data_all_almost/total*100,1)
        result[config] += [percentage_dsq, percentage_dsq_almost, percentage_lpips, percentage_lpips_almost, percentage_playa, percentage_playa_almost, percentage_all, percentage_all_almost]
    

    template_fill = "& {0} & {1}\\% & {2}\\% & {3}\\% & {4}\\% & {5}\\% & {6}\\% & {7}\\% & {8}\\% \\\\\n"
    

    template = ""
    for config in sorted_configs:
        template += template_fill.format(config, *result[config])
    print(template)

# 5, 0, 3
#count_percentage_good(3, data)
import random

def get_good_levels(data):

    BASELINE_DSC = 0.863
    BASELINE_LPIPS = 0.167
    BASELINE_PLAYA = 0.93

    good_levels = []
    decent_levels = []
    okay_levels = []

    for elem in data:
        
        dsc = elem[-3]
        lpips = elem[-2]
        playa = elem[-1]

        if dsc <= BASELINE_DSC and lpips >= BASELINE_LPIPS and playa >= BASELINE_PLAYA:
            good_levels.append(elem)
        if dsc <= BASELINE_DSC * 1.05 and lpips >= BASELINE_LPIPS * 0.95 and playa >= BASELINE_PLAYA * 0.95:
            decent_levels.append(elem)
        if dsc <= BASELINE_DSC * 1.1 and lpips >= BASELINE_LPIPS * 0.9 and playa >= BASELINE_PLAYA * 0.9:
            okay_levels.append(elem)
    
    print("Levels with good metrics:", len(good_levels))
    print("Levels with decent metrics:", len(decent_levels))
    print("Levels with okay'ish metrics:", len(okay_levels))

    if good_levels:
        print("Here is a random GOOD level:", random.choice(good_levels))
    elif decent_levels:
        print("Here is a random DECENT level:", random.choice(decent_levels))
    elif okay_levels:
        print("Here is a random OKAY'ISH level:", random.choice(okay_levels))
    else:
        print("All levels are bad")

get_good_levels(data)
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_theme(style="whitegrid")
#sns.color_palette("Spectral", as_cmap=True)

#sns.relplot(x="x", y="y", hue="split", data=dict_with_data, col="col", palette="deep")

#plt.figure()
#plt.scatter(list(map(lambda x: x[7], data)),
#            list(map(lambda x: x[8], data)))
#plt.figure()
#plt.scatter(list(map(lambda x: x[7], good_data_loss)),
#            list(map(lambda x: x[8], good_data_loss)))
#plt.figure()
#plt.scatter(list(map(lambda x: x[7], good_data_sat)),
 #           list(map(lambda x: x[8], good_data_sat)))
#plt.figure()
#plt.scatter(list(map(lambda x: x[7], good_data_div)),
#            list(map(lambda x: x[8], good_data_div)))
#plt.show()