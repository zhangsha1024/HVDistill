CONFIG=$1

export OMP_NUM_THREADS=12; python pretrain_twostage.py --cfg_file ${CONFIG} 