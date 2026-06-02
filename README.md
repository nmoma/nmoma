<h1 align="center"> Primitive-based Truncated Diffusion for Efficient Trajectory Generation of Differential Drive Mobile Manipulators </h1>

<div align="center">
  
[[Website]](https://nmoma.github.io/nmoma/)

<!-- <img src="static/images/ip.png" style="height:50px;" /> -->
<!-- <img src="static/images/meta.png" style="height:50px;" /> -->
</div>

## Code

Code will be released in stages:

- [x] **Pretrained checkpoints + evaluation pipelines with ROS**  

- [ ] **Dataset and code for training and testing**  

- [ ] **Code for data generation pipelines**

You are welcomed to open issues related to this repository.

## Quickstart

We strongly recommend using our docker image:

```
docker pull nmoma/nmoma
```

inside the container:
```
cd /workspace

git clone https://github.com/nmoma/nmoma.git

hf download nmoma/nmoma \
  --repo-type dataset \
  --local-dir .

catkin_make

./auto.sh ckpt_anchor_ptrans_attention_exp # replace with desired experiment
```
## Manual Setup

Alternatively, you can pull the code and download the checkpoints with:
```
git clone https://github.com/nmoma/nmoma.git

hf download nmoma/nmoma \
  --repo-type dataset \
  --local-dir ./nmoma
```

### Install Pointcept

Download `Pointcept` from `https://github.com/Pointcept/Pointcept/tree/df36980119f4636beb2d02d04ef3b2fec0fddfba`

Then:

```
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install tqdm pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

cd libs/pointops
python setup.py install
cd ../..

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu118  # choose version match your local cuda version

# Open3D (visualization, optional)
pip install open3d

sudo apt install ros-noetic-vrpn-client-ros libasio-dev ros-noetic-ompl
sudo apt-get install ros-noetic-rosfmt libglfw3-dev libglew-dev libdw-dev
sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
pip3 install empy==3.3.4
```

### Install h5py

download source code of hdf5-1.12.3 from `https://hdf-wordpress-1.s3.amazonaws.com/wp-content/uploads/manual/HDF5/HDF5_1_12_3/src/hdf5-1.12.3.tar.gz`

Add the paths to file `~/.bashrc`

```
export CC=mpicc
export HDF5_MPI="ON"
export HDF5_DIR="/path/to/parallel/hdf5"  #! If this isn't found by default
export PATH="$HDF5_DIR/bin:$PATH"
```

Then:

```
tar -zxvf hdf5-1.12.3.tar.gz
cd hdf5-1.12.3
./configure --enable-parallel --enable-shared
make
sudo make install
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ --no-binary h5py --no-cache h5py
```

### Build the project

```
catkin_make
```