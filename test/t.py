import re

pattern = r'saved-model-(\d+)-([1-9]\d*\.\d*|0\.\d*[1-9]\d*)-([1-9]\d*\.\d*|0\.\d*[1-9]\d*).hdf5'
prog = re.compile(pattern)

str = 'saved-model-0-0.78-0.16.hdf5'

matchObj = prog.match(str)

c_epoch = matchObj.group(1)
c_loss = matchObj.group(2)
c_acc = matchObj.group(3)

pass
