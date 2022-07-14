import os
# assign directory
directory = './copypaste/eval_output/setup2/'
 
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
        assert len(fd) == 5
        fd1 = list(map(lambda x: int(x[1:]), fd[:-1]))
        fd2 = float(fd[-1][2:])
        fd = fd1 + [fd2]

        with open(f, 'r') as datafile:
            lines = datafile.readlines()[11:]
            measurements = list(map(lambda x: float(x.split(" ")[2]), lines))
            fd += measurements
        
        data.append(fd)

good_data_loss = sorted(data, key = lambda x: x[5])[:len(data)//10]    
print(len(good_data_loss))
good_data_sat = sorted(good_data_loss, key = lambda x: x[8], reverse=True)[:len(good_data_loss)//3]    
print(len(good_data_sat))
good_data_div = sorted(good_data_sat, key = lambda x: x[7], reverse=True)[:len(good_data_sat)//4]    
print(len(good_data_div))
print(good_data_div)
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(list(map(lambda x: x[7], data)),
            list(map(lambda x: x[8], data)))
plt.figure()
plt.scatter(list(map(lambda x: x[7], good_data_loss)),
            list(map(lambda x: x[8], good_data_loss)))
plt.figure()
plt.scatter(list(map(lambda x: x[7], good_data_sat)),
            list(map(lambda x: x[8], good_data_sat)))
plt.figure()
plt.scatter(list(map(lambda x: x[7], good_data_div)),
            list(map(lambda x: x[8], good_data_div)))
plt.show()