import argparse
# python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--data_dir', type = str, default = 'flowers/', 
                        help = 'path to the folder of flower images')
    parser.add_argument('--learning_rate', type = float, default = 0.03, 
                        help = 'learning rate for the model')
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                        help = 'CNN Architecture to use for training')
    parser.add_argument('--hidden_units', type = int, default = 1024, 
                        help = 'number of hidden layers')
    parser.add_argument('--epochs', type = int, default = 3, 
                        help = 'number of epochs')
    
    in_args = parser.parse_args() 
    
    return in_args
