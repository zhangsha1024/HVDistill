CONFIG=$1

export OMP_NUM_THREADS=12; python downstream.py --cfg_file ${CONFIG} 