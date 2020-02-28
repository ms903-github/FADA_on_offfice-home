CLASSNUM=65
CLASS_INIT=0
CSV_PATH="csv/csv65"
CONFIG_PATH="./config/cfg1"
DEVICE=0

set -eu
echo data_augmentation...
python utils/transform.py
echo making_csv...
python utils/csv_maker.py --dataset_dir="./dataset/RealWorld" --save_dir=$CSV_PATH --class_num=$CLASSNUM --domain_name="RealWorld" --class_init=$CLASS_INIT
python utils/csv_maker.py --dataset_dir="./dataset/Product" --save_dir=$CSV_PATH --class_num=$CLASSNUM --domain_name="Product" --class_init=$CLASS_INIT
echo done.
echo sampling...
python utils/sampler.py --class_num=$CLASSNUM --class_init=$CLASS_INIT --csv_path=$CSV_PATH
echo done.
echo making_pair...
python utils/pair_maker.py --class_num=$CLASSNUM --class_init=$CLASS_INIT --csv_path=$CSV_PATH
python utils/adv_pair_maker.py --class_num=$CLASSNUM --class_init=$CLASS_INIT --csv_path=$CSV_PATH
echo done.
echo start_training...
env CUDA_VISIBLE_DEVICES=$DEVICE python train.py $CONFIG_PATH --offset=$CLASS_INIT --csv_path=$CSV_PATH