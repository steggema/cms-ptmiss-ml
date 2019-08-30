
'''Create a virtual dataset from input files
'''

import optparse
import h5py

# configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--inputs', dest='inputs',
                  help='input files (comma separated)', default='tree_100k.h5', type='string')
parser.add_option('-o', '--output', dest='output',
                  help='output file', default='virtual.h5', type='string')
parser.add_option('-d', '--datasets', dest='datasets',
                  help='Names of datasets (comma separated)', default='X,X_x,Y,Z', type='string')
(opt, args) = parser.parse_args()

inputs = opt.inputs.split(',')
datasets = opt.datasets.split(',')



layouts = {}

for dataset in datasets:
    shapes = []
    dtypes = []
    for inp in inputs:
        with h5py.File(inp, 'r') as f_in:
            shapes.append(f_in[dataset].shape)
            dtypes.append(f_in[dataset].dtype)

    shape = (sum(sh[0] for sh in shapes),) + shapes[0][1:]

    # Assemble virtual dataset
    layout = h5py.VirtualLayout(shape=(shape), dtype=dtypes[0]) # , dtype='i4'

    offset = 0
    for i, inp in enumerate(inputs):
        length = shapes[i][0]
        vsource = h5py.VirtualSource(inp, dataset, shape=shapes[i])
        layout[offset:offset + length] = vsource
        offset += length

    layouts[dataset] = layout

# Add virtual dataset to output file
with h5py.File(opt.output, 'w', libver='latest') as f:
    for dataset in datasets:
        f.create_virtual_dataset(dataset, layouts[dataset], fillvalue=0)
