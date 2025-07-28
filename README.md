# Spiking Heterogenous Graph Attention Network (SpikingHAN)
%% This repository is for SpikingHAN model by [@Qian Peng](https://github.com/QianPeng369).%% 

## Requirements	

 - Python == 3.8.16
 - dgl-cu116 == 0.9.1
 - numpy == 1.24.4
 - pandas == 2.0.3
 - scipy == 1.10.1
 - torch == 1.13.1+cu116
 - pynvml == 11.5.3
## Reproduction
 - DBLP
```python
python main.py --dataset DBLP --seed 1 --training_rate 0.2 --lr 0.005 --hidden_units 32 --dropout1 0.6 --dropout2 0.5 --T 9 --alpha 2.0 --tau 1.0 --surrogate sigmoid --neuron PLIF --reset subtract --threshold 0.05
```
 - ACM
```python
python main.py --dataset ACM --seed 261 --training_rate 0.2 --lr 0.005 --hidden_units 64 --dropout1 0.4 --dropout2 0.4 --T 40 --alpha 2.0 --tau 1.0 --surrogate sigmoid --neuron PLIF --reset subtract --threshold 0.5
```
 - IMDB
```python
python main.py --dataset IMDB --seed 52 --training_rate 0.2 --lr 0.005 --hidden_units 32 --dropout1 0.3 --dropout2 0.4 --T 13 --alpha 2.0 --tau 1.0 --surrogate sigmoid --neuron PLIF --reset subtract --threshold 0.5
```
