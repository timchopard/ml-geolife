if [ -z "$(ls -A data/_raw)" ]; then
    echo ":: Downloading raw data to data/_raw"
    cd data/_raw
    kaggle competitions download -c geolifeclef-2024
    cd -
else
    echo ":: data/_raw is not empty. Aborting download."
fi
