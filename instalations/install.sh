echo "====================================================="
echo "Installing pytorch, torchvision and cudatoolkit"
echo "====================================================="

conda install -y pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge
python -c 'import torch;print(torch.__version__);print(torch.version.cuda)'

echo "====================================================="
echo "Installing mmcv, mmdet and other dependencies"
echo "====================================================="

pip install --no-input dicttoxml albumentations terminaltables imagecorruptions funcy scikit-learn pycocotools wget
pip install --no-input -U openmim
mim install mmengine
mim install "mmcv==1.3.17"
mim install "mmdet==2.28.2"
pip install yapf==0.40.1

echo "====================================================="
echo "Installation completed!"
echo "====================================================="