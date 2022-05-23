from copy import deepcopy
import json
import sys

def switch_wall_monster(level):
    switched_level = deepcopy(level)
    for i in range(len(level)):
        for j in range(len(level[0])):
            if switched_level[i][j] == 1:
                switched_level[i][j] = 2
            elif switched_level[i][j] == 2:
                switched_level[i][j] = 1
    return switched_level

def main():
    load_path = sys.argv[1]
    save_path = sys.argv[2]
    switched_levels = []
    with open(load_path, 'r') as level_output:
        for line in level_output:
            list_level = json.loads(line)
            switched_level = switch_wall_monster(list_level)
            switched_levels.append(json.dumps(switched_level))
    

    with open(save_path, 'w') as edited_level_output:
        edited_level_output.writelines(map(lambda x: x + "\n", switched_levels))

main()