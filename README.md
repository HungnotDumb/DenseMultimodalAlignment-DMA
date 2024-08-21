
# Dense Multimodal Alignment for Open-Vocabulary 3D Scene Understanding, ECCV2024

### [Paper](https://arxiv.org/abs/2407.09781) 

Welcome! This is the official implementation of the DMA work.

## Abstract
Recent vision-language pre-training models have shown remarkable generalization ability in zero-shot recognition tasks. Our work mainly focuses on Open-Vocabulary 3D scene understanding methods. We have developed a DMA (Dense Multimodal Alignment) framework to co-embed various modalities into a common space focusing on maximizing their mutual benefits. This work employs a dual-path integration procedure combining frozen CLIP visual features and learnable mask features without compromising the open-vocabulary skills.

![framework](doc/framework_2.jpg)

## Updates
- **2024.07.02**: Our DMA project is accepted by ECCV 2024!

## Data Preparation 

Please follow the structure to arrange the folders:
```
└── data
    ├── ...
└── output
    ├── ...
```
## Evaluation
In-depth evaluation instructions for different datasets are shared in this section.

#Scannet

```bash
python -u evaluate.py   --config=./config/scannet/ours_openseg.yaml   feature_type distill  save_folder ./save
```

#Qualitative Results

![overview](doc/visualization.jpg)

## Citation
If you find this project useful, please cite our work:

    @inproceedings{li2024dense,
        author = {...},
        title = {Dense Multimodal Alignment for Open-Vocabulary 3D Scene Understanding},
        booktitle = {ECCV},
        year = {2024}
    }

## Acknowledgement
This project is based on [OpenScene](https://github.com/pengsongyou/openscene). Thanks for their excellent work.