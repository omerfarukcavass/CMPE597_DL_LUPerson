The repository is for the CMPE597 Deep Learning course project. The base project is chosen as [Unsupervised Pre-training for Person Re-identification](https://arxiv.org/abs/2012.03753).
Original project: [LUPerson](https://github.com/DengpanFu/LUPerson-NL)

## LUPerson Dataset
In the original project LUPerson dataset is used. LUPerson is currently the largest unlabeled dataset for Person Re-identification, which is used for Unsupervised Pre-training. LUPerson consists of 4M images of over 200K identities and covers a much diverse range of capturing environments. 

**Details can be found at ./LUP**.

## Market1501 Dataset
In this implementation experiments are performed on Market1501 dataset. It contains 1501 identities which are captured by six different cameras, and 32,668 pedestrian image bounding-boxes obtained using the Deformable Part Models pedestrian detector. Each person has 3.6 images on average at each viewpoint. The dataset is split into two parts: 750 identities are utilized for training and the remaining 751 identities are used for testing. 90% and 10% of the training set is used in the pretraining stage and the finetuning stage, respectively. 
[Scalable Person Re-Identification: A Benchmark](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html)

## Pre-trained Models
| Model | path |
| :------: | :------: |
| ResNet50 Default pretrained | [R50](https://drive.google.com/file/d/1pFyAdt9BOZCtzaLiE-W3CsX_kgWABKK6/view?usp=sharing) |

## Finetuned Results

|Dataset | mAP | cmc1 | path |
|:------:|:---:|:----:|:----:|
| Def_SSL + ft | 31.4 | 55.0 | [Model](https://drive.google.com/file/d/1bV27gwAsX8L3a3yhLoxAJueqrGmQTodV/view?usp=sharing) |.
| CJ_SSL + ft | 32.0 | 56.1 | [Model](https://drive.google.com/file/d/1leUezGnwFu8LKG2N8Ifd2Ii9utlJU5g4/view?usp=sharing) |
| AF_SSL + ft | 31.1 | 55.0 | [Model](https://drive.google.com/file/d/1AlXgY5bI0Lj7HClfNsl3RR8uPi2nq6Zn/view?usp=sharing) |
| PT + Def_ft | 48.6 | 74.2 | [Model](https://drive.google.com/file/d/1BQ-zeEgZPud77OtliM9md8Z2lTz11HNh/view?usp=sharing)|
| PT + CJ_ft | 48.2 | 73.5 | [Model](https://drive.google.com/file/d/1BQ-zeEgZPud77OtliM9md8Z2lTz11HNh/view?usp=sharing)|
| PT + AF_ft | 46.4 | 72.4 | [Model](https://drive.google.com/file/d/1BQ-zeEgZPud77OtliM9md8Z2lTz11HNh/view?usp=sharing)|

These numbers are a little different from those reported in our paper, and most are slightly better.



## Citation

```
@article{fu2020unsupervised,
  title={Unsupervised Pre-training for Person Re-identification},
  author={Fu, Dengpan and Chen, Dongdong and Bao, Jianmin and Yang, Hao and Yuan, Lu and Zhang, Lei and Li, Houqiang and Chen, Dong},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2021}
}
```
