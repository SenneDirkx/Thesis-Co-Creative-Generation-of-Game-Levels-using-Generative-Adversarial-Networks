import os
from typing import Dict, List, Tuple

def main() -> None:
    DIRECTORY_PATH = './original'
    directory = os.fsencode(DIRECTORY_PATH)
    file_id = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and filename != "README.txt": 
            parse_level(DIRECTORY_PATH + '/' + filename, file_id)
            file_id += 1

def parse_level(path: str, file_id: int) -> None:
    TARGET_PATH = './parsed'
    levels: Dict[Tuple[int], List[str]] = {}
    with open(path, 'r') as level:
        lines = list(map(lambda x: x.strip('\n'), level.readlines()))
        for li in range(len(lines)):
            line = lines[li]
            for ci in range(len(line)):
                y = li//16
                x = ci//11
                key = (x, y)
                if key not in levels:
                    levels[key] = []
                if len(levels[key]) == (li % 16):
                    levels[key].append("")
                ch = line[ci]
                levels[key][li % 16] += ch
    
    for levi in levels:
        result = levels[levi]
        unique_chars = set()
        for rl in result:
            for rc in rl:
                unique_chars.add(rc)
        unique_chars.discard('-')
        if len(unique_chars) == 0:
            continue
        level = map(lambda x: x + '\n', result)
        with open(TARGET_PATH + '/' + f"lev{file_id}_{levi[0]}_{levi[1]}.txt", 'w') as target:
            target.writelines(level)

if __name__ == "__main__":
    main()