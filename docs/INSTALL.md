## Installation
- install the virtual environment and pytorch:
  ```
  conda create --name env_name python==3.8
  source activate env_name
  pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```

- install cmake: `conda install cmake`

- install sparse conv: `pip install spconv-cu113`

- install Waymo evaluation module: `pip install waymo-open-dataset-tf-2-2-0`

- install the requirements of DGMFusion: `cd DGMFusion && pip install -r requirements.txt`

- install the requirements of image_modules: `cd DGMFusion/detection/al3d_det/models/image_modules/swin_model && pip install -r requirements.txt && python setup.py develop`

- compile DGMFusion:
  ```
  cd DGMFusion/utils && python setup.py develop
  ```
- compile the specific algorithm module:
  ```
  cd DGMFusion/detection  && python setup.py develop
  ```
- compile the specific dcn module:
  ```
  cd DGMFusion/detection/al3d_det/models/ops  && python setup.py develop
  ```
