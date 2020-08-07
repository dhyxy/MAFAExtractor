# MAFAExtractor

This is a tool for extracting data from the [MAFA Dataset](https://openaccess.thecvf.com/content_cvpr_2017/html/Ge_Detecting_Masked_Faces_CVPR_2017_paper.html). It succesfully extracts all labels and data from the dataset's provided MATLAB files into a Pandas DataFrame.

The primary function is `extract_mafa()` which is all you really need if you're just extracting the data into Pandas. If the filename isn't the original `LabelTrainAll.mat` or `LabelTestAll.mat`, then you have to provide the `dataset_type` which can be either *"train"* or *"test"*. You can also choose whether you want the dataframe to be cleaned to have more readable and processed values by the `clean` parameter (which by default is True), or can be set to False if you  require the dataset's original headings.

## Usage:
```python3
from mafaextractor import extract_mafa

df = extract_mafa("path/to/LabelTrainAll.mat <or> LabelTestAll.mat")

# differing file names
df = extract_mafa("path/to/IChangedTheTestingSetsFileName.mat", dataset_type="test")

# no cleaning
df = extract_mafa("path/to/TestingSet.mat", dataset_type="test", clean=False)
```
If you run into any bugs or have any concerns feel free to contact me via e-mail at dhyeyl1@outlook.com!

## License
This project is licensed under the terms of the MIT license.
