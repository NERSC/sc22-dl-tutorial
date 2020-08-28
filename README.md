3D U-Net testing with amp

Simple demonstrator to profile performance of 3D U-Net with and without AMP enabled. Trains on random data.

To run, just do `python train.py`, use option `--config=base` for the baseline fp32 setup and `--config=withAMP` for AMP.

Testing with ngc-20.07 and ngc0.08 images, I get only a 1.2x speedup using AMP.


