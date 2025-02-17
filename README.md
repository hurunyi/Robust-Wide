# [ECCV2024] Robust-Wide: Robust Watermarking Against Instruction-Driven Image Editing
Official implementation of [Robust-Wide: Robust Watermarking Against Instruction-Driven Image Editing](https://arxiv.org/abs/2402.12688).
## Train

1. Download the [data](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) and put them into the data dir ./data.

2. Configure the train script and then run it.

```
bash train.sh
```

## Inference
1. Put your original image in ./examples.

2. Download the [checkpoints](https://drive.google.com/drive/folders/1Y67UuFQiWqX5mA_1TBUs9FB4OUvazrZe?usp=drive_link) and put them in ./checkpoints.

3. Configure the inference script and then run it.
```
bash inference.sh
```

## Acknowledgements
This code builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library.

## Cite
If you find this repository helpful, please cite the following:
```
@inproceedings{hu2025robust,
  title={Robust-wide: Robust watermarking against instruction-driven image editing},
  author={Hu, Runyi and Zhang, Jie and Xu, Ting and Li, Jiwei and Zhang, Tianwei},
  booktitle={European Conference on Computer Vision},
  pages={20--37},
  year={2025},
  organization={Springer}
}
```
