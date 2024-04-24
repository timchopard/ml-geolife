ba_flag=''
tf_flag=''
pt_flag=''

while getopts 'btp' flag; do
    case "${flag}" in
        b)  ba_flag="true" ;;
        t)  tf_flag="true" ;;
        p)  pt_flag="true" ;;
        *)  echo "exiting" && exit 1 ;;
    esac
done

if [ $ba_flag -eq "true" ]; then
    echo ":: [INSTALL] Base packages" ;
    conda activate geolife ;
    conda install anaconda::numpy anaconda::matplotlib anaconda::ipykernel conda-forge::pandas conda-forge::tqdm conda-forge::opencv
fi

if [ $tf_flag -eq "true" ]; then
    echo ":: [INSTALL] Tensorflow" ;
    conda activate geolife ;
    conda install conda-forge::tensorflow ;
fi

if [ $pt_flag -eq "true" ]; then
    echo ":: [INSTALL] PyTorch" ;
    conda activate geolife ;
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia ;
fi


