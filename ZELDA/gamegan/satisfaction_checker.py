import json
import sys

OUTPUT_FILENAME = sys.argv[1]  

def is_surrounded_by_walls(level):
    for y in range(len(level)):
        for x in range(len(level[y])):
            if y in {0, 15}:
                if level[y][x] != 1:
                    return False
            elif y in {1, 14}:
                if x in range(4,7):
                    if level[y][x] != 1 and level[y][x] != 3:
                        return False
                else:
                    if level[y][x] != 1:
                        return False
            else:
                if x in {1, 9}:
                    if level[y][x] != 1 and level[y][x] != 3:
                        return False
                elif x in {0, 10}:
                    if level[y][x] != 1:
                        return False

    return True

def has_doors(level):
    if level[1][4:7] == [3, 3, 3]:
        return True
    if level[14][4:7] == [3, 3, 3]:
        return True
    if level[7][1] == 3 and level[8][1] == 3:
        return True
    if level[7][9] == 3 and level[8][9] == 3:
        return True
    return False

def has_reasonable_enemies(level):
    enemy_counter = 0
    for y in range(2,len(level)-2):
        for x in range(2,len(level[y])-2):
            if level[y][x] == 2:
                enemy_counter += 1
            
    return enemy_counter <= 63

def is_all_reachable(level):
    floor_counter = 0
    first_found = False
    first_pos = None
    for y in range(2,len(level)-2):
        for x in range(2,len(level[y])-2):
            if level[y][x] == 0:
                if not first_found:
                    first_found = True
                    first_pos = (x, y)
                floor_counter += 1

    reachable_counter = 0
    visited = set()        
    queue = [first_pos]

    while len(queue) > 0:
        current = queue.pop(0)
        if current in visited:
            continue
        reachable_counter += 1
        visited.add(current)
        options = [(current[0]-1, current[1]), (current[0]+1, current[1]), (current[0], current[1]-1), (current[0], current[1]+1)]
        for opt in options:
            if opt[0] < 0 or opt[0] > 10 or opt[1] < 0 or opt[1] > 15:
                continue
            if opt in visited:
                continue
            if level[opt[1]][opt[0]] != 0:
                continue
            queue.append(opt)
    return 0.9 * floor_counter < reachable_counter

def is_satisfactory(level):
    #if not (is_surrounded_by_walls(level) and has_doors(level) and has_reasonable_enemies(level) and is_all_reachable(level)):
    #    print(is_surrounded_by_walls(level), has_doors(level), has_reasonable_enemies(level), is_all_reachable(level))
    return is_surrounded_by_walls(level) and has_doors(level) and has_reasonable_enemies(level) and is_all_reachable(level)

def main():
    levels = []
    with open(OUTPUT_FILENAME, 'r') as level_output:
        for line in level_output:
            list_level = json.loads(line)
            levels.append(list_level)
    
    total = 0
    passed = 0
    for i in range(len(levels)):
        if is_satisfactory(levels[i]):
            passed += 1
        
        total += 1
    score = passed / total
    return score


score = main()
print(score)