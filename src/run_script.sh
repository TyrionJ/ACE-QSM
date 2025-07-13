# to exe folder
cd execution


# preprocess
python preprocess.py -D 1


# train dwt_diffusion
python diffusion_train.py -D 1 -T Task001_DWTDiffusion -denoiser DWTDenoiserNet -sintr 5000 -iters 600000 -devices 0 --c


# train normal unet
python normal_train.py -D 1 -T Task002_BaseUNet -N BaseUNet -b 4 -d 2 -e 200 --c


# diffusion infer
python diffusion_infer.py \
-data_dir "../../data/Infer_samples" \
-task_dir "../../data/ACE-QSM_results/Dataset001_iLSQR/Task001_DWTDiffusion" \
-sub_name "ALL" -method "iLSQR" -device 1
