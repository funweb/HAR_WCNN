import re, os


pattern = r'[Pbest_]?saved-model-.*-(\d+)-(\d*\.\d*)-(\d*\.\d*).hdf5'

s = 'saved-model-20220201_cairo_WCNN_rms_1000_9999_0-000100-0.013441-0.848958.hdf5'
prog = re.compile(pattern)
matchObj = prog.match(s)
c_epoch = matchObj.group(1)
c_loss = matchObj.group(2)
c_acc = matchObj.group(3)

pass

