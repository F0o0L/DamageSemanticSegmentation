#conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/win-64/
conda create --name MyTorch python=3.7
conda activate MyTorch
#conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2
#conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1
pip install opencv-python
pip install opencv-contrib-python
pip install albumentations
pip install pandas
pip install tqdm
pip install labelme
#pip install imgviz
pip install --upgrade tensorflow
conda install -c conda-forge jupyterlab
jupyter-notebook --generate-config
pip install ipykernel
python -m ipykernel install --user --name=torch
conda clean --tarballs
conda clean -y --all
