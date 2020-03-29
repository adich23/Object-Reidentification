## Run command
python siamese_two_stream.py train

## Test resuls on [Vehicle Reid dataset](https://arxiv.org/abs/1911.05541)
* **Combined Results** - 
  * **Precision:94.342  | Recall:88.326 | F-score:90.99**

* 5-fold test run results 
  * P:87.10 R:97.56 F:92.03 A:96.36 | tp: 29470, tn: 105599, fp: 4364, fn: 737 
  * P:98.23 R:87.18 F:92.38 A:97.14 | tp: 31028, tn: 143004, fp: 559, fn: 4562 
  * P:95.42 R:77.89 F:85.77 A:94.89 | tp: 28763, tn: 148592, fp: 1380, fn: 8167 
  * P:92.98 R:85.95 F:89.32 A:95.78 | tp: 31869, tn: 140994, fp: 2407, fn: 5211 
  * P:97.98 R:93.05 F:95.45 A:98.12 | tp: 31532, tn: 125678, fp: 651, fn: 2355 
  
## Model Architecture and input example
![Current Model](https://github.com/adich23/Object-Reidentification/blob/master/two-stream-model.png)
