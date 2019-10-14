# README



## 1. clusters for digit

output for clusters_for_digit.py：

```
n_digits: 10, 	 n_samples 1797, 	 n_features 64
__________________________________________________________________________________
init		time	inertia	homo	compl	v-meas	ARI-	AMI-	silhouette
k-means++	0.17s	69676	0.683	0.722	0.702	0.572	0.699	0.140
random   	0.12s	69415	0.604	0.652	0.627	0.467	0.623	0.147
PCA-based	0.03s	70769	0.669	0.696	0.683	0.558	0.679	0.147
__________________________________________________________________________________

__________________________________________________________________________________
clusters		time	inertia	nor-	homo	comp
MB_KMeans		0.02s	3034	0.444	0.437	0.452
AffinityPro		6.44s	0	0.458	0.476	0.441
MeanShift		0.77s	0	0.445	0.460	0.431
Spectral 		0.22s	0	0.461	0.459	0.462
Ward     		0.08s	0	0.451	0.444	0.458
Agglom   		0.08s	0	0.053	0.010	0.279
DBSCAN   		0.02s	0	0.308	0.316	0.301
__________________________________________________________________________________
```

![myplot](myplot.png)

output for figures_of_digits.py:

![myplot0](myplot0.png)



## clusters for document

