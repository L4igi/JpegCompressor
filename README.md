# JPEG Compressor using Python 
A JPEG compression algorithm was created using Python 3. MatPlotLib is used as a help library to read in images as matrices and display those. In
addition Numpy is used for matirx manipulation and calculations. Pictures have to be in JPEG format to work with this program. The DCT works with
8x8 pixel blocks, therefore images with a resolution not a multiple of eight have to be padded with additional pixels which are removed again in the last step.
## Why it was created?
This JPEG compressor was implemeted as part of the NTM course at the University of Vienna. It showcases the JPEG compression process, from scratch, step by step, using only
MatPlotLib for File read/write. This implementation does not aim for performance but for readability and simplicity. 
To showcase different compression outcomes, multiple subsampling Methods and downsampling tables can be chosen. 

## How to use 
1. Input a file name in the current directory (has to end in .jpg)
2. choose number between 1 and 5 to select subsampling method
3. choose high or low compression degree (1 || 2)
## Additional Packages 
MatPlotLib: https://matplotlib.org/

NumPy: https://numpy.org/
