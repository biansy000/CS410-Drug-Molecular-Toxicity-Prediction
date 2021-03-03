# CS410-Drug-Molecular-Toxicity-Prediction

The code is for the AI class in SJTU, which aims at predicting the toxication of drug molecules.

The model mainly uses GCNConv [[1]](#1) to predict the result. 
You can run the model using ``python main.py``.Model parameters can be configured in ``all_cfg.py`` (only temporarily used, and many options do not actually function properly, which will be removed later).

Moreover, models in ``model.py`` are not actually used, because they have inferior performance in the given dataset than GCNConv. may make them also available from ``main.py`` later ...

## References
<a id="1">[1]</a> Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks[J]. arXiv preprint arXiv:1609.02907, 2016.
