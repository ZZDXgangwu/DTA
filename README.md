# Drug-target affinity prediction based on topological enhanced graph neural networks
### Usage
1. Edit config.py

2. Run the script `python run.py`


### Data
* Davis and KIBA：[https://github.com/hkmztrk/DeepDTA/tree/master/data](https://github.com/hkmztrk/DeepDTA/tree/master/data "https://github.com/hkmztrk/DeepDTA/tree/master/data")
* Protein sequence：[UniProt](https://www.uniprot.org/ "UniProt")
* Proteins cavities features： [CaviDB](https://www.cavidb.org/ "CaviDB") and [Fpocket](https://github.com/Discngine/fpocket)
### Data preprocess
* preprocess script：[scripts.py](https://github.com/595693085/DGraphDTA/blob/master/scripts.py "here")
* pockets json：A dict mapping Cavity IDs to lists of residue positions extracted from [CaviDB](https://www.cavidb.org/ "CaviDB") and [Fpocket](https://github.com/Discngine/fpocket)
