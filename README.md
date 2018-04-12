## Setup

Please ensure the following are installed and accessible by the directory containing the program files (the libraries may be installed using pip):
* Python3
    * Music21
    * Keras
    * Tensorflow
    * h5py
    
## To Train the LSTM:

Simply run **mcuai_lstm.py** from the command line.

Ex:
    `python3 ./mcuai_lstm.py`
    
## Once the Weights are Generated:

You may run `compose.py` file and the network will generate a midi file stored in the `/output` directory.
