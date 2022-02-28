import torch
import os

ONE_HOT_CODE = {
    'F': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'M': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'P': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'O': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'I': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'D': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'S': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'W': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

def main() -> None:
    DIRECTORY_PATH = './parsed'
    directory = os.fsencode(DIRECTORY_PATH)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"): 
            tensorize_level(DIRECTORY_PATH + '/' + filename, filename[:-4])
        

def tensorize_level(path: str, file_id: str) -> None:
    # TARGET_PATH = './tensorizedConv'

    # with open(path, 'r') as level:
    #     lines = list(map(lambda x: x.strip('\n'), level.readlines()))
    #     dim10 = []
    #     for i in range(10):
    #         dim16 = []
    #         for j in range(len(lines)):
    #             dim11 = []
    #             for k in range(len(lines[j])):
    #                 block = lines[j][k]
    #                 one_hot = ONE_HOT_CODE[block]
    #                 one_hot_val = one_hot[i]
    #                 dim11.append(one_hot_val)
    #             dim16.append(dim11)
    #         dim10.append(dim16)
    #     tensor = torch.tensor(dim10)
    #     torch.save(tensor, TARGET_PATH + '/' + file_id + '.pt')
    TARGET_PATH = './tensorized'
    with open(path, 'r') as level:
        lines = list(map(lambda x: x.strip('\n'), level.readlines()))
        one_hot_elems = [list(map(lambda c: ONE_HOT_CODE[c], list(string))) for string in lines]
        one_hot_lines = list(map(lambda t: [item for sublist in t for item in sublist], one_hot_elems))
        tensor = torch.tensor(one_hot_lines)
        torch.save(tensor, TARGET_PATH + '/' + file_id + '.pt')

if __name__ == "__main__":
    main()