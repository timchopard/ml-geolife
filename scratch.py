import sys

from datamanagement.dataloader import DataLoader

# Run the data loader with system arguments:
# --read        or  -r  :   Read in the data     
# --elevation   or  -e  :   Read in the data and calculate elevation
def main():
    args = sys.argv[1:]
    data = None
    read_list = ['--read', '-r', '--elevation', '-e']
    path_train = "data/GLC24_PA_metadata_train.csv"
    path_test = "data/GLC24_PA_metadata_test.csv"
    if sum(1 for _ in [i for i in args if i in read_list]) > 0:
        data = DataLoader(path_train, None)
    if '--elevation' in args or '-e' in args:
        data._get_elevations()
    
    if '--image' in args or '-i' in args:
        data = DataLoader(None, None)
        print(data._get_image_from_id(212, rgb=True, train=True))

    if '--full' in args or '-f' in args:
        data = DataLoader(train=path_train, test=None)
        data.preprocess_csv()

    if '--save-arrays' in args:
        arg_vals = [True, False]
        data = DataLoader(train=path_train, test=path_test)
        for value_data in arg_vals:
            for value_color in arg_vals:
                location = data.images_to_numpy(value_data, value_color)
                print(f":: [SAVED] {location}")
        

if __name__ == "__main__":
    main()
