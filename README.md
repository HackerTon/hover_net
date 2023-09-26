![](docs/conic_banner.png)


## Set Up Environment

1. `conda env create -f environment.yml`
2. `conda activate hovernet`
3. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

# Training HoVer-Net for CoNIC Challenge

This branch is dedicated to training the HoVer-Net for the [CoNIC challenge](https://conic-challenge.grand-challenge.org/). All parameters are hard-coded and are expected to work out of the box, as long as users follow the preparation steps mentioned below:

1. Setup the enviroment as instructed above.
2. Download [Pytorch ImageNet ResNet50](https://download.pytorch.org/models/resnet50-0676ba61.pth) and put
it under `exp_output/local/`.
3. Download the data from the [CoNIC challenge](https://conic-challenge.grand-challenge.org/)
4. Extract the data into the `exp_output/local/data` folder.
5. Run `python generate_split.py` to generate a number
of training and validation subsets. For the baseline in the
challenge, we use the `FOLD_IDX=0`.
6. Run `python run.py --gpu 0`

# To inferencing

1. Run `infer.ipynb` using vscode.
2. Run all the code blocks inside of `infer.ipynb`.

## Hyper-parameters

The following files contain the hyper-paramters for training the HoVer-Net
- `models/hovernet/net_desc.py`: Define the HoVer-Net architecture. Unlike the original paper, we use the ResNet50 from pytorch as backbone and padded convolution in the decoders (resulting in the same output size as the input).
- `models/hovernet/opt.py`: Define the arguments for HoVer-Net training phases. If you want to modify the number of training epochs, modify it here. You can also find the weights for each loss component here.
- `dataloader/train_loader.py`: `FileLoader` defines how the
images are loaded and pre-processed (this include generating ground-truth from annotation). Compared with original version, we turn off all affine transformation (defined in
`__get_augmentation`).
- `param/template.yaml`: This file contains runtime parameters to input to the architecture or the running loop
(`batch_size`).

## Citation

If any part of this code is used, please give appropriate citation to our HoVer-Net paper and our [CoNIC challenge](https://conic-challenge.grand-challenge.org/). <br />

BibTex entry: <br />
```
@article{graham2019hover,
  title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
  author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak, Jin Tae and Rajpoot, Nasir},
  journal={Medical Image Analysis},
  pages={101563},
  year={2019},
  publisher={Elsevier}
}

@article{graham2021conic,
  title={CoNIC: Colon Nuclei Identification and Counting Challenge 2022},
  author={Graham, Simon and Jahanifar, Mostafa and Vu, Quoc Dang and Hadjigeorghiou, Giorgos and Leech, Thomas and Snead, David and Raza, Shan E Ahmed and Minhas, Fayyaz and Rajpoot, Nasir},
  journal={arXiv preprint arXiv:2111.14485},
  year={2021}
}
```

## Authors

* [Quoc Dang Vu](https://github.com/vqdang)
* [Simon Graham](https://github.com/simongraham)
