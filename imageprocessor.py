import os
import sys
from datamanagement.dataloader import DataLoader 

PATH_TRAIN = "data/GLC24_PA_metadata_train.csv"
PATH_TEST = "data/GLC24_PA_metadata_test.csv"

def process(loc):    
    try:
        os.mkdir(loc)
    except FileExistsError:
        pass
    arg_vals = [True, False]
    data = DataLoader(train=PATH_TRAIN, test=PATH_TEST)
    for value_data in arg_vals:
        for value_color in arg_vals:
            location = data.images_to_numpy(value_data, value_color, loc)
            print(f":: [SAVED] {location}")

def main():
    args = sys.argv[1:]
    save_loc = "./processed_images/"
    if len(args) == 0:
        print(f"To save to the default location of: [{save_loc}] "      \
            + "run this command again with the \'-d\' flag\n"           \
            + "To enter a custom path use the \'-c \"[custom_path]\"")
        return 

    if args[0] == '-c':
        save_loc = args[1]
        if save_loc[-1] != '/': save_loc += '/'
    print(f"Save directory is:    {save_loc}")
    confirmation = input("Is this correct? (y/n):\t")
    while True:
        match confirmation.lower():
            case 'n':
                return 
            case 'y':
                process(save_loc)
                return
            case _:
                print("Please enter only \'y\' or \'n\'")

if __name__ == "__main__":
    main()