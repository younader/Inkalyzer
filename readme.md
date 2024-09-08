# Vesuvius Ink XAI <img align="center" width="60" height="60" src="chatgpt_logo.png">

<!-- ![Vesuvius Challenge XAI](chatgpt_logo.png) -->

The Repository contains an XAI toolkit for analyzing ink detection and explaining predictions using various information bottleneck techniques from XAI literature. 

The toolkit is designed to work with any type of models as the bottleneck is placed outside of the network (i.e in input space).    

This repo builds on the findings of the First Letters and Grand prize of 2023. 




![3d Attribution](attribution.gif)


# 🚀 Get Started

I recommend using a docker image like `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel` for your development environment. Kaggle/Colab images should work fine as well. 

To install this project, run:

```bash
pip install -r requirements.txt

```

You can download this checkpoint (which is an old I3D I had laying around): [link](https://drive.google.com/file/d/1BXvADKzJZ4ZSDHsKXa-fZES3MOfMrMdJ/view?usp=sharing), or you can use any checkpoint you want (refer to externals.models for more details). 

# Documentation
The repo implements different variation of IBA and diffmask 

| Method Name       | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `animate_history`       | animates the volume layers importance throughout the optimization run                               |
| `attribute_z_diffmask` | Generates depth attribution using Diffmask approach                       |
| `attribute_z_with_diffmask_pooling`     | Similar to Diffmask pooling with additional pooling to smooth out the depth probabilities.                 |
| `attribute_z_lowres`   | A downsampled version for IBA for smoother attribution    |
| `attribute_z`         | The base IBA method for generating attributions along the z axis  |
| `attribute_3d`        | IBA based 3d attribution, performs attribution along z first, then attributes windows in the x-y.                            |

<!-- # IBA GUI

To run the GUI:

```bash
streamlit run GUI.py
```
 -->
