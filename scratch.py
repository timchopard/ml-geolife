import sys

from datamanagement.dataloader import DataLoader

# Run the data loader with system arguments:
# --read        or  -r  :   Read in the data     
# --elevation   or  -e  :   Read in the data and calculate elevation
def main():
    args = sys.argv[1:]
    print(args)
    data = None
    read_list = ['--read', '-r', '--elevation', '-e']
    if sum(1 for _ in [i for i in args if i in read_list]) > 0:
        path_train = "data/GLC24_PA_metadata_train.csv"
        data = DataLoader(path_train, None)
    if '--elevation' in args or '-e' in args:
        data._get_elevations()
        

if __name__ == "__main__":
    main()
