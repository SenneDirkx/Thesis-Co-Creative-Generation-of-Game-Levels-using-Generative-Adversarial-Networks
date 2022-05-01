CHANNELS = 10
HEIGHT = 16
WIDTH = 11

store = {}

def create_options(h, w):
    if (h == 0 or h == 15):
        return [8]
    elif (w == 0 or w == 10):
        return [8]
    elif (h == 1 or h == 14) and (w < 4 or w > 6):
        return [8]
    elif (h == 1 or h == 14) and (4 <= w <= 6):
        return [6, 8]
    elif (w == 1 or w == 9) and (h < 7 or h > 8):
        return [8]
    elif (w == 1 or w == 9) and (7 <= h <= 8):
        return [6, 8]
    else:
        return [0, 1, 2, 3, 4, 5, 7, 9]


for h in range(HEIGHT):
    for w in range(WIDTH):
        options = create_options(h, w)
        store[(h,w)] = options

var_id = 1
rules = []
for i in range(CHANNELS):
    for j in range(HEIGHT):
        for k in range(WIDTH):
            possibilities = store[(j, k)]
            #possible_var_ids = list(range(var_id, var_id+10))
            #var_ids = []
            #for i in possibilities:
            #    var_ids.append(str(possible_var_ids[i]))
            
            if i != 6 and i not in possibilities:
                rules.append(f"-{var_id} 0\n")
            
            var_id += 1

with open(f"./final/constraint_total.txt", 'w') as output:
    output.write(f"p cnf {var_id-1} {len(rules)}\n")
    output.writelines(rules)