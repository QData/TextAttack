
.. _installation:


Installation
==============

To use TextAttack, you must be running Python 3.6+. Tensorflow needs to be installed for users, and Java needs to be installed for developers. A CUDA-compatible GPU is optional but will greatly improve speed. To install, simply run::

    pip install textattack 

You're now all set to use TextAttack! Try running an attack from the command line::

    textattack attack --recipe textfooler --model bert-base-uncased-mr --num-examples 10

This will run an attack using the TextFooler_ recipe, attacking BERT fine-tuned on the MR dataset. It will attack the first 10 samples. Once everything downloads and starts running, you should see attack results print to ``stdout``.

Read on for more information on TextAttack, including how to use it from a Python script (``import textattack``).

Using a CUDA-compatible GPU
---------------------------------------
To improve speed, use a CUDA-compatible GPU. The following instructions install CUDA, cuDNN, TensorFlow, PyTorch, and TextAttack via three different methods with no prior CUDA software needed. The first method installs the software via ``conda`` from Anaconda. The second method installs the software directly from each source repository. The third method covered uses CUDA enabled containers with Docker.

1. Installing CUDA and TextAttack via Anaconda
----------------------------------------------

1. Install CUDA Toolkit and CUDA 11.2 (more detailed instructions available here: `docs.nvidia.com <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`__). The below commands are for Ubuntu 20.04.

.. code-block:: console

    $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

    $ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

    $ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

    $ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

    $ sudo apt-get update

    $ sudo apt-get -y install cuda build-essential

    $ echo "export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}" >> ~/.bashrc

2. Install Anaconda (Miniconda3 in this case)

.. code-block:: console

    $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

    $ bash miniconda.sh

Follow the installation prompt, then restart.

3. Create env for TextAttack

.. code-block:: console

    $ conda create -yn textattack-env

    $ conda activate textattack-env

    $ conda install -y cudatoolkit tensorflow

4. Install TextAttack

.. code-block:: console

    $ pip install textattack

    $ textattack attack --recipe textfooler --model bert-base-uncased-mr --num-examples 10

5. Verify CUDA is being used

.. code-block:: console
    
    $ python3 -c "import torch; print(torch.cuda.is_available())"


2. Installing CUDA and TextAttack directly
----------------------------------------------

1. Install CUDA Toolkit and CUDA 11.2 (more detailed instructions available here: `docs.nvidia.com <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`__). The below commands are for Ubuntu 20.04.

.. code-block:: console

    $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

    $ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

    $ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

    $ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

    $ sudo apt-get update

    $ sudo apt-get -y install cuda build-essential

    $ echo "export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}" >> ~/.bashrc

2. Reboot system and verify installation

.. code-block:: console

    $ nvidia-smi

    $ nvcc --version
    
3. Install cuDNN (more detailed instructions available here: `docs.nvidia.com <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux>`__)
    
The installation of cuDNN requires an NVIDIA developer account to access the download. Go to https://developer.nvidia.com/cudnn and follow the instructions to download cuDNN. For this guide, download the following:

- `cuDNN Runtime Library for Ubuntu20.04 x86_64 (Deb)`
- `cuDNN Code Samples and User Guide for Ubuntu20.04 x86_64 (Deb)`

.. code-block:: console

    $ sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb

    $ sudo dpkg -i libcudnn8-samples_8.1.1.33-1+cuda11.2_amd64.deb

To verify the installation

.. code-block:: console

    $ cp -r /usr/src/cudnn_samples_v8/ $HOME

    $ cd  $HOME/cudnn_samples_v8/mnistCUDNN

    $ make clean && make

    $ ./mnistCUDNN

4. Install TextAttack

.. code-block:: console
    
    $ python3 -m venv textattack-env

    $ source textattack-env/bin/activate

    $ pip3 install tensorflow, textattack

    $ textattack attack --recipe textfooler --model bert-base-uncased-mr --num-examples 10

5. Verify CUDA is being used

.. code-block:: console

    $ python3 -c "import torch; print(torch.cuda.is_available())"


3. Docker with NVIDIA Container Toolkit to use CUDA enabled containers
-----------------------------------------------------------------------

0. Prerequisites - have the NVIDIA GPU driver installed

1. Install Docker Engine (more detailed instructions available here: `docs.docker.com <https://docs.docker.com/engine/install/ubuntu/>`__)

.. code-block:: console

    $ sudo apt-get update

    $ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    $ echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    $ sudo apt-get update

    $ sudo apt-get install docker-ce docker-ce-cli containerd.io

2. Install NVIDIA Container Toolkit (more detailed instructions available here: `docs.nvidia.com <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit>`__)

.. code-block:: console

    $ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
        && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
        && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    $ sudo apt-get update

    $ sudo apt-get install -y nvidia-docker2

    $ sudo systemctl restart docker

3. Verify that Docker Engine and NVIDIA Container Toolkit installed properly

.. code-block:: console

    $ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

4. Install TextAttack

For this installation guide, the TensorFlow GPU container is used as a base.

.. code-block:: console

    $ sudo docker run --gpus all -it tensorflow/tensorflow:latest-gpu

This starts the container and opens a ``bash`` shell within. Inside the container: 

.. code-block:: console

    # pip install textattack

    # textattack attack --recipe textfooler --model bert-base-uncased-mr --num-examples 10


5. Verify CUDA is being used

.. code-block:: console

    # python3 -c "import torch; print(torch.cuda.is_available())"

.. _TextFooler: https://arxiv.org/abs/1907.11932
