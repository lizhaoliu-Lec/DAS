DATA_ROOT = <ENTER_YOUR_DATA_ROOT_HERE>

#################################### CUB ####################################
# ===================> MS + IBN^{512} <===================
python main.py --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --gpu 4 --bs 112 --embed_dim 512 --samples_per_class 2 --loss multisimilarity --arch bninception_double_frozen_normalize_multiSim_300epoch_200_250_embeddingProducer_DAS_numScale8 --tau 200 250 --embedding_producer_used --produce_type DAS --das_dfs_num_scale 8 --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --das_mts_scale 0.01 --detach
# ===================> MS + R^{512} <===================
python main.py --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --gpu 0 --bs 180 --embed_dim 512 --samples_per_class 2 --loss multisimilarity --decay 0.0008 --arch resnet50_double_frozen_normalize_multiSim_300epoch_200_250_embeddingProducer_DAS_numScale8 --tau 200 250 --embedding_producer_used --produce_type DAS --das_dfs_num_scale 8 --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --das_mts_scale 0.01 --detach

#################################### CARS ####################################
# ===================> MS + IBN^{512} <===================
python main.py --dataset cars196 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --gpu 3 --bs 112 --embed_dim 512 --samples_per_class 2 --loss multisimilarity --arch bninception_double_frozen_normalize_multiSim_300epoch_200_250_embeddingProducer_DAS_numScale8 --tau 200 250 --embedding_producer_used --produce_type DAS --das_dfs_num_scale 8 --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --das_mts_scale 0.01 --detach
# ===================> MS + R^{512} <===================
python main.py --dataset cub200 --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --gpu 3 --bs 112 --embed_dim 512 --samples_per_class 2 --loss multisimilarity --arch resnet50_double_frozen_normalize_multiSim_300epoch_200_250_embeddingProducer_DAS_numScale8 --tau 200 250 --embedding_producer_used --produce_type DAS --das_dfs_num_scale 8 --das_dfs_scale_left 0.99 --das_dfs_scale_right 1.01 --das_mts_scale 0.01 --detach

#################################### SOP ####################################
# ===================> MS + IBN^{512} <===================
python main.py --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --gpu 2 --bs 112 --embed_dim 512 --samples_per_class 2 --loss multisimilarity --arch bninception_double_frozen_normalize_multiSim_300epoch_200_250_embeddingProducer_DAS_numScale8 --embedding_producer_used --produce_type DAS --das_dfs_num_scale 8 --das_mts_scale 0.1 --no_train_metrics
# ===================> MS + R^{512} <===================
python main.py --dataset online_products --kernels 6 --source $DATA_ROOT --n_epochs 300 --tau 200 250 --seed 0 --gpu 7 --bs 112 --embed_dim 512 --samples_per_class 2 --loss multisimilarity --arch resnet50_double_frozen_normalize_multiSim_300epoch_200_250_embeddingProducer_DAS_numScale8 --embedding_producer_used --produce_type DAS --das_dfs_num_scale 8 --das_mts_scale 0.1 --no_train_metrics

