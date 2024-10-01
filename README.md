**Combining Transformer with a Depth Map for enhanced Low Light Performance for UHD Images**

By- Dhanush Adithya,  Harshil Bhojwani.

Download Depthanythingv2 metric depth from 
https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true 
into **"metric_depth/checkpoints/"**

Download model from
https://drive.google.com/drive/folders/1AZchyGAR5vtQpIEZTdDYH9t605nk_YBR
into **checkpoints/LLFormer_LOL/models/**

**OS - Linux**

Dataset Download Link for LOL - https://daooshee.github.io/BMVC2018website/
within dataset directory, check the directory heirarchy configuration inside the directory

run 
python test.py
all the configs have been set to run the file and change test.py if any of the above-mentioned directories are changed

for training
python train.py
all the configs have been set to run the file and change yaml file in configs if any of the above-mentioned directories are changed

Files edited train.py and test.py, files added metric-depth/pipeline.py




References
LLformer - https://github.com/TaoWangzj/LLFormer/tree/main
Metric Depth - https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth


