# Wasserstein For Documents
Implementation for our ECIR 2018 paper "Fast Cross-lingual Document Retrieval using Regularized Wasserstein Distance".
The code in the repository implements the `Wass` and `Entro_Wass` models of our paper. 
The implementations heavily rely on: 
- [Python Optimal Transport (POT)](https://github.com/rflamary/POT) for calculating the wasserstein distances
- [Scikit-learn](http://scikit-learn.org/stable/)  for extending the nearest neighbors classifier to provide a generic framework for using out method.

# Running the code
To run the code, one first needs to download the [Numberbatch](https://github.com/commonsense/conceptnet-numberbatch) embeddings we used in this paper. We provide a script to download the embeddings and extract those for a subset of languages, e.g., English and French. To do that, first clone the repository, move the the directory and execute the script:

```
git clone https://github.com/balikasg/WassersteinRetrieval
cd WassersteinRetrieval
bash get_Embeddings.sh
```
This will take some time, and it will output informative messages for its progress. It will create two files: `concept_net_1706.300.en` and `concept_net_1706.300.fr`, the containing the English and French word embeddings respectively. 

To run the cross-lingual retrieval experiments, run:
```
python emd.py concept_net_1706.300.en concept_net_1706.300.fr wiki_data/wikicomp.enfr.2k.en wiki_data/wikicomp.enfr.2k.fr 500 french
```
This runs the `emd.py` program with several arguments. The first two stand for the embeddings, the second two for the datasets where retrieval is performed, the fifth for the upper limit of words to be kept for each document (we used 500 for efficiency) and the last one for the second language (by default the first is english).


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
The code can be parallelized easily as one needs to calculate the distance of each query document with every document in the set of available documents.
We have used `pathos` to parellize the calculations in the level of queries. Having *N* queries will send them to the available cores. In the figure below we illustrate the performance 
benefits when parallelizing the example of the section "Running the code", using 1,2,6,10,14,18 and 22 cores in a Intel(R) Xeon(R) CPU E5-2643 v3 @ 3.40GHz machine. 
Notice that `Entro_Wass` needs more time, but the difference is small when having more than 10 cores available. Also,`Entro_Wass` can be implemented with GPUs, but we did not have access to one while writing the paper. 
  
![Timings with cores](https://github.com/balikasg/WassersteinForDocuments/blob/master/timing.png)
