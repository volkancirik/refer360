pip install tensorboardX
pip install nltk
pip install -U torch torchvision
pip install git+https://github.com/facebookresearch/fvcore.git
pip install Cython
pip install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI
git clone https://github.com/airsplay/py-bottom-up-attention.git py_bottom_up_attention
cd py_bottom_up_attention; pip install -r requirements.txt; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'; python setup.py build develop; cd -
git clone https://github.com/airsplay/lxmert.git; cd lxmert; pip install -r requirements.txt; cd -

pip uninstall torch torchvision -y
conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch -y
conda install -c conda-forge opencv=4.1.0 -y
conda install tqdm -y
conda install -c anaconda scipy -y
conda install -c conda-forge spacy -y
conda install seaborn -y
conda install scikit-learn -y
python -m spacy download en_core_web_sm
python -m ipykernel install --user --name refer360 --display-name "refer360"
