# TeCM-CLIP: Text-based Controllable Multi-attribute Face Image Manipulation

Our framework supports **hairstyle, hair color, emotion, gender and age editing**.
Select the corresponding attribute mapper and enter the text prompt.

## Getting Started
### Dependencies
```bash
$ Python >= 3.7 (Recommend to use Anaconda or Miniconda)
$ PyTorch >= 1.7
$ CLIP
$ Option: NVIDIA GPU + CUDA
```
### Installation
```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

### Models
Please download the pre-trained model from the following link. We only provide the pretrained model of Hairstyle here.
| Path | Description
| :--- | :----------
|[Hairstyle](https://drive.google.com/file/d/19L8pn2ae_boD3MjCkEevJHkTiEX_DVb3/view?usp=sharing)  | Our pre-trained Hairstyle model.  
|Hair Color  | Our pre-trained Hair Color model.  
|Emotion  | Our pre-trained Emotion model.  
|Gender  | Our pre-trained Gender model.  
|Age  | Our pre-trained Age model.  

If you wish to use the pretrained model for training or inference, you may do so using the flag `--checkpoint_path`.  
### Auxiliary Models and Latent Codes
In addition, we provide various auxiliary models and latent codes inverted by [e4e](https://github.com/omertov/encoder4editing) needed for training your own model from scratch.
| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://drive.google.com/file/d/1pts5tkfAcWrg4TpLDu6ILF5wHID32Nzm/view?usp=sharing) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
|[IR-SE50 Model](https://drive.google.com/file/d/1FS2V756j-4kWduGxfir55cMni5mZvBTv/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during HairCLIP training.
|[Train Set](https://drive.google.com/file/d/1gof8kYc_gDLUT4wQlmUdAtPnQIlCO26q/view?usp=sharing) | CelebA-HQ train set latent codes inverted by [e4e](https://github.com/omertov/encoder4editing).
|[Test Set](https://drive.google.com/file/d/1j7RIfmrCoisxx3t-r-KC02Qc8barBecr/view?usp=sharing) | CelebA-HQ test set latent codes inverted by [e4e](https://github.com/omertov/encoder4editing).  

By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`.
## Training
### Training TeCM-CLIP
The main training script can be found in `scripts/train.py`.   
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.  
#### **Training the TeCM-CLIP Mapper**
Example of Using Text to Edit Hairstyle
```bash
cd mapper
python scripts/train.py \
--exp_dir=/path/to/experiment \
--change_type="HairStyle" \
--description="description/hairstyle_description.txt" \
--latents_train_path=/path/to/train_faces.pt \
--latents_test_path=/path/to/test_faces.pt \
--batch_size=1  \
--id_lambda=0.2 \
--clip_lambda=0.6 \
--latent_l2_lambda=1.0 \
--face_l2_lambda=2.0 \
--bg_l2_lambda=0.0 \
--learning_rate=0.0001 \
--mapper_type='LevelsMapper'
```

## Testing
### Inference
The main inference script can be found in `inference.py`. Inference results are saved to `test_opts.exp_dir`.  
#### Example of Using Text to Edit Hairstyle
```bash
cd mapper
python inference.py \
--exp_dir=/path/to/experiment \
--checkpoint_path=./pretrained_models/hairclip.pt \
--latents_test_path=/path/to/test_faces.pt \
--description="short curly hair"
```


## Acknowledgements
This code is based on [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) and [HairCLIP](https://github.com/wty-ustc/HairCLIP).


