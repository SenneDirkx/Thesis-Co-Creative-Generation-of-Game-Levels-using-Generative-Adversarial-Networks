from copy import deepcopy
import json
import random
import sys

NB_SAMPLES = int(sys.argv[1])

def dice_score(level1, level2):
    level1_d = get_dummies(level1)
    level2_d = get_dummies(level2)

    a = 0
    b = 0
    #c = 0

    for i in range(len(level1_d)):
        for j in range(len(level1_d[0])):
            for k in range(len(level1_d[0][0])):
                if level1_d[i][j][k] == level2_d[i][j][k] and level1_d[i][j][k] == 1:
                    a += 1
                elif level1_d[i][j][k] != level2_d[i][j][k]:
                    b += 1
    
    result = (2*a)/(2*a + b)
    return result        

def get_dummies(level):
    dummy_level = deepcopy(level)
    for i in range(len(level)):
        for j in range(len(level[0])):
            if level[i][j] == 0:
                dummy_level[i][j] = [1, 0, 0, 0]
            elif level[i][j] == 1:
                dummy_level[i][j] = [0, 1, 0, 0]
            elif level[i][j] == 2:
                dummy_level[i][j] = [0, 0, 1, 0]
            elif level[i][j] == 3:
                dummy_level[i][j] = [0, 0, 0, 1]
            else:
                raise ValueError("MORE THAN 4 CLASSES ???")
    return dummy_level

def main():
    levels = []
    with open('level_output.json', 'r') as level_output:
        for line in level_output:
            list_level = json.loads(line)
            levels.append(list_level)
    
    total_score = 0

    # sample and measure
    for _ in range(NB_SAMPLES):
        selected1 = random.randint(0, len(levels)-1)
        selected2 = random.randint(0, len(levels)-1)
        score = dice_score(levels[selected1], levels[selected2])
        total_score += score

    avg_score = total_score / NB_SAMPLES
    return avg_score


print(main())
