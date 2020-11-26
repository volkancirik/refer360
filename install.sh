conda create -n refer360 python=3.7 -y
source activate refer360
conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch -y
conda install -c conda-forge opencv=4.1.0 -y
conda install tqdm -y
conda install -c conda-forge spacy -y
python -m spacy download en_core_web_sm
pip install tensorboardX
pip install nltk
pip install -U torch torchvision
pip install git+https://github.com/facebookresearch/fvcore.git
git clone https://github.com/airsplay/py-bottom-up-attention.git py_bottom_up_attention
cd py_bottom_up_attention; pip install -r requirements.txt; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'; python setup.py build develop
pip install Cython
pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI
