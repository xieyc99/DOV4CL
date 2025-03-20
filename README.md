## Code
### 1. Pre-train Encoders
Using SimCLR, BYOL, SimSiam or MoCo v3 to pre-train encoders:
```
python main_train.py \
    --dataset <Dataset Name> \
    --data_path <PATH/TO/DATASET/DIR> \
    --arch <Encoder Architecture> \
    --method <Self-supervised Algorithm> \
    --part <Use either a random half, the remaining half, or the entire dataset for training.> \
    --ratio 0.5 \
    --num_workers 8 \
    --gpu <GPU ID>
```

Using DINO to pre-train encoders:
```
python dino_train.py \
    --dataset <Dataset Name> \
    --data_path <PATH/TO/DATASET/DIR> \
    --arch <Encoder Architecture> \
    --part <Use either a random half, the remaining half, or the entire dataset for training.> \
    --num_workers 8 \
    --gpu <GPU ID>
```

### 2. Dataset Ownership Verification by Our Method
DOV4CL.py: The code of our method.
You can run it in the following way:
```
python DOV4CL.py \
    --D_public <Public Dataset of Defender> \
    --M_shadow_arch <Encoder Architecture of M_shadow> \
    --M_shadow_dataset <the Training Set of M_shadow> \
    --M_shadow_path <PATH/TO/M_shadow/DIR> \
    --M_suspect_arch <Encoder Architecture of M_suspect> \
    --M_suspect_dataset <the Training Set of M_suspect> \
    --M_suspect_path <PATH/TO/M_suspect/DIR> \
    --n_sample_train <k_public> \
    --n_sample_test <k_private> \
    --n_aug <M> \
    --n_aug_local <N> \
    --n_epoch <K> \
    --lamda <a> \
    --gpu <GPU ID>
```

## Acknowledgement
This resipotry is based on [CTRL]((https://github.com/meet-cjli/CTRL)

## Citation
If you find our data or project useful in your research, please cite:
```
@article{xie2025dataset,
  title={Dataset Ownership Verification in Contrastive Pre-trained Models},
  author={Xie, Yuechen and Song, Jie and Xue, Mengqi and Zhang, Haofei and Wang, Xingen and Hu, Bingde and Chen, Genlang and Song, Mingli},
  journal={arXiv preprint arXiv:2502.07276},
  year={2025}
}
```
