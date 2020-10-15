set -euo pipefail

# Install NodeJS + Tensorboard + TensorboardX

curl -sL https://deb.nodesource.com/setup_14.x | bash - \
    && apt-get install -y nodejs

apt-get update && apt-get install -y --no-install-recommends libsndfile1-dev

python3 -m pip install --no-cache --upgrade \
    soundfile==0.10.2 \
    ipywidgets==7.5.1 \
    jupyter_tensorboard==0.2.0 \
    widgetsnbextension==3.5.1 \
    tensorboard==2.1.1 \
    tensorboardX==2.1 --user
jupyter labextension install jupyterlab_tensorboard \
   && jupyter nbextension enable --py widgetsnbextension \
   && jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Revise default shell to /bin/bash
jupyter notebook --generate-config \
  && echo "c.NotebookApp.terminado_settings = { 'shell_command': ['/bin/bash'] }" >> /root/.jupyter/jupyter_notebook_config.py
