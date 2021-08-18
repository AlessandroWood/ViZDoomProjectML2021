# Progetto ViZDoom

Autori: Luca Gregori & Alessandro Wood  
Corso: Machine Learning

Si veda il notebook __Relazione Finale__

## Preparazione dell'ambiente di sviluppo


- Installazione driver Nvidia (proprietari) (prerequisiti per cuda):

```
sudo apt install nvidia-driver-460
```

- Installazione Cuda (versione 11.2):
```
sudo apt install nvidia-cuda-toolkit
```
- Installazione nvidia cuDNN dal seguente [link](https://developer.nvidia.com/rdp/cudnn-download) (richiede registrazione al programma nvidia developer)  
cuDNN Runtime Library for Ubuntu20.04 x86_64 (Deb) è il pacchetto da installare


- Esportazioni variabili d’ambiente:
```
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
- Per verificare l’installazione di cuda
```
nvcc -V
```
- Installazione di Tensorflow/PyTorch tramite pip

- Installazione ViZDoom
si segua la guida quick start per python: [VizDoom](https://github.com/mwydmuch/ViZDoom).

## Riferimenti


Arnold :
@inproceedings{chaplot2017arnold,
  title={Arnold: An Autonomous Agent to Play FPS Games.},
  author={Chaplot, Devendra Singh and Lample, Guillaume},
  booktitle={Proceedings of AAAI},
  year={2017},
  Note={Best Demo award}
}


ViZDoom:
@article{wydmuch2018vizdoom,
  title={ViZDoom Competitions: Playing Doom from Pixels},
  author={Wydmuch, Marek and Kempka, Micha{\l} and Ja{\'s}kowski, Wojciech},
  journal={IEEE Transactions on Games},
  year={2018},
  publisher={IEEE}
Rarity of Events:
@article{roe,
title={Automated Curriculum Learning by Rewarding
Temporally Rare Events},
author={Niels Justesen, Sebastian Risi},
year={2018}

Automated Curriculum Learning by Rewarding Temporally Rare Events | Niels Justesen & Sebastian Risi | [link](https://arxiv.org/pdf/1803.07131.pdf)

Human-level control through deep reinforcement
learning | DeepMind | 26 february 2015 | vol 518 | Nature | [link](https://www.nature.com/articles/nature14236)