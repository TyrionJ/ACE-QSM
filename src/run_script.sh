# preprocess
python preprocess.py -D 1


# train dwt_diffusion
# iLSQR
python execution/diffusion_train.py -D 1 -T Task001_DWTDiffusion -denoiser DWTDenoiserNet -sintr 5000 -iters 600000 -devices 0 --c
# QSMnet
python execution/diffusion_train.py -D 12 -task Task001_DWTDiffusion -denoiser DWTDenoiserNet -sintr 5000 -iters 600000 -devices 3 --c
# msQSM
python execution/diffusion_train.py -D 13 -task Task001_DWTDiffusion -denoiser DWTDenoiserNet -sintr 5000 -iters 600000 -devices 3 --c
# INR-QSM
python execution/diffusion_train.py -D 14 -task Task001_DWTDiffusion -denoiser DWTDenoiserNet -sintr 5000 -iters 600000 -devices 3 --c
# STAR-QSM
python execution/diffusion_train.py -D 15 -task Task001_DWTDiffusion -denoiser DWTDenoiserNet -sintr 5000 -iters 600000 -devices 1 --c


python execution/diffusion_train.py -D 11 -task Task002_BaseDiffusion -denoiser BaseDenoiserNet -sintr 5000 -iters 600000 -devices 3 --c

# train normal unet
python execution/normal_train.py -D 11 -T Task003_BaseUNet -N BaseUNet -b 4 -d 2 -e 200 --c



# T=200
python execution/diffusion_train.py -D 11 -task Task004_DWTDiffusion_T-200 -denoiser DWTDenoiserNet -sintr 5000 -steps 200 -iters  120000 -devices 0 --c
# T=500
python execution/diffusion_train.py -D 11 -task Task005_DWTDiffusion_T-500 -denoiser DWTDenoiserNet -sintr 5000 -steps 500 -iters  300000 -devices 1 --c


# diffusion testing
python execution/diffusion_testing.py \
-dataset "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_raw/Dataset011_iLSQR_e-3" \
-task_dir "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_results/Dataset011_iLSQR_e-3/Task004_DWTDiffusion_T-200" \
-store_dir "DW-T200_preds" -batch 8 -device 0 -data_key "sTE_0075"

python execution/diffusion_testing.py \
-dataset "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_raw/Dataset011_iLSQR_e-3" \
-task_dir "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_results/Dataset011_iLSQR_e-3/Task002_BaseDiffusion" \
-store_dir "base_preds" -batch 8 -device 1 -data_key "test" --reverse


python execution/diffusion_testing.py \
-dataset "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_raw/Dataset012_QSMnet_e-3" \
-task_dir "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_results/Dataset012_QSMnet_e-3/Task001_DWTDiffusion" \
-store_dir "dwt_preds" -batch 8 -device 2 -data_key "test"

python execution/diffusion_testing.py \
-dataset "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_raw/Dataset013_msQSM_e-3" \
-task_dir "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_results/Dataset013_msQSM_e-3/Task001_DWTDiffusion" \
-store_dir "dwt_preds" -batch 8 -device 0 -data_key "test"

python execution/diffusion_testing.py \
-dataset "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_raw/Dataset014_INR-QSM_e-3" \
-task_dir "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_results/Dataset014_INR-QSM_e-3/Task001_DWTDiffusion" \
-store_dir "dwt_preds" -batch 8 -device 3 -data_key "test"

python execution/diffusion_testing.py \
-dataset "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_raw/Dataset015_STAR-QSM_e-3" \
-task_dir "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_results/Dataset015_STAR-QSM_e-3/Task001_DWTDiffusion" \
-store_dir "dwt_preds" -batch 8 -device 2 -data_key "test"


# diffusion infer
python execution/diffusion_infer.py \
-data_dir "/remote-home/hejj/Data/Researches/MRI/Multiple Sclerosis/Fr_Xu_Rui" \
-task_dir "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_results/Dataset015_STAR-QSM_e-3/Task001_DWTDiffusion" \
-sub_name "03_Hanbing,07_LiuYiChuan,08_WangQingRu,09_LiXueYuan,15_LuoXiaoLi_f_21,16_YangQuanHong_m_27,21_PanQiTao_26,22_wuyuanxun_m_36,23_YangRuiZhi_f_33" \
-method "STAR-QSM" -device 3

python execution/diffusion_infer.py \
-data_dir "/remote-home/hejj/Data/Researches/MRI/microbleeds" \
-task_dir "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_results/Dataset011_iLSQR_e-3/Task001_DWTDiffusion" \
-sub_name "ALL" -method "iLSQR" -device 0

python execution/diffusion_infer.py \
-data_dir "/remote-home/hejj/Data/Researches/MRI/drug_addition/N" \
-task_dir "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_results/Dataset015_STAR-QSM_e-3/Task001_DWTDiffusion" \
-sub_name "CHEN_DING_ZHEN" \
-method "STAR-QSM" -device 3

python execution/diffusion_infer.py \
-data_dir "/remote-home/hejj/Data/Researches/MRI/ACE-QSM-data/GE 750w" \
-task_dir "/remote-home/hejj/Data/runtime/ACE-QSM/ACE-QSM_results/Dataset011_iLSQR_e-3/Task001_DWTDiffusion" \
-sub_name "ALL" -method "iLSQR" -device 1

python diffusion_infer.py \
-data_dir "../../data/Infer_samples" \
-task_dir "../../data//ACE-QSM_results/Dataset001_iLSQR/Task001_DWTDiffusion" \
-sub_name "ALL" -method "iLSQR" -device 1
