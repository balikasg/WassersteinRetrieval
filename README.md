# Wasserstein For Documents
Implementation for our ECIR 2018 paper "Fast Cross-lingual Document Retrieval using Regularized Wasserstein Distance".
The code in the repository implements the `Wass` and `Entro_Wass` models of our paper. 
The implementations heavily rely on: 
- [Python Optimal Transport (POT)](https://github.com/rflamary/POT) for calculating the wasserstein distances
- [Scikit-learn](http://scikit-learn.org/stable/)  for extending the nearest neighbors classifier to provide a generic framework for using out method.



# Citing
In case you use the model or the provided code, please cite our paper:
```
@InProceedings{balikas2018ecir,
  author    = {Georgios Balikas and Charlotte Laclau and Ievgen Redko and Massih-Reza Amini},
  title     = {Fast Cross-lingual Document Retrieval using Regularized Wasserstein Distance},
  booktitle = {Proceedings of the 40th European Conference {ECIR} conference on Information Retrieval, {ECIR} 2018, Grenoble, France, March 26-29, 2018},
  year      = {2018}}
```





# Timings
![Timings with cores](https://github.com/balikasg/WassersteinForDocuments/blob/master/timing.png)
