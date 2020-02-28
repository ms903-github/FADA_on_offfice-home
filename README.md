# FADA_on_offfice-home
pytorch 1.0 implementation on Few-shot adversarial domain adaptation[1] on office-home[2] dataset

## about
This repository is unofficial implementation of Few-shot adversarial domain adaptation[1] on office-home dataset.   
Though [1] treats digit recognition task and office dataset task, we applied the algorythm to office-home dataset which has more number of classes.  

## usage
1. Make `dataset` directory, and put the office-home dataset under it. Dataset should be structured like
```
dataset  
|  
L-RealWorld  
| L Alarm_clock    
| L backpack  
| ...  
L- Product    
|  L Alarm_clock  
|  L backpack  
...   
```
2. run   
```
mkdir result
mkdir csv
bash script.sh

```
