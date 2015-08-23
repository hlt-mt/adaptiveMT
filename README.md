# Adaptive MT server

Clone of adaptive [MT server](http://mt4cat.fbk.eu/software/adaptive-mt-server) with added tuning procedures.

This package supports tuning of dense and sparse features with Pairwise Ranking Optimization algorithm. 
Add these lines in your configuration file and tuning will automatically start. 
```
[tuning]
switch=on
learning_rate= 0.01
```
By default the learning rate is set to 0.01
