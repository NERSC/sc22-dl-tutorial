import argparse
import os

import h5py
import numpy as np

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input_file', type=str, required=True, help='hdf5 file to convert')
  parser.add_argument('-d', '--data_field', type=str, required=True, help='name of data field in hdf5 file')
  parser.add_argument('-l', '--label_field', type=str, required=True, help='name of label field in hdf5 file')

  args = parser.parse_args()

  output_prefix = os.path.splitext(os.path.basename(args.input_file))[0]

  f = h5py.File(args.input_file, 'r')

  # Load data field and save in dhwc format
  data = f[args.data_field]
  data = np.transpose(data, axes=[1,2,3,0])
  np.save(output_prefix + "_data.npy", data)

  del data

  # Load label field and save in dhwc format
  label = f[args.label_field]
  label = np.transpose(label, axes=[1,2,3,0])
  np.save(output_prefix + "_label.npy", label)

  del label

if __name__ == '__main__':
  main()
