# FishFSRNet
Code for paper "Super-Resolving Face Image by Facial Parsing Information"

![image](https://github.com/wcy-cs/FishFSRNet/blob/main/fishfsrnet.png)

## Citation 
```
@ARTICLE{10090424,
  author={Wang, Chenyang and Jiang, Junjun and Zhong, Zhiwei and Zhai, Deming and Liu, Xianming},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science}, 
  title={Super-Resolving Face Image by Facial Parsing Information}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TBIOM.2023.3264223}}
```
 
## Results
 [BaiDu](https://pan.baidu.com/s/1wll2RrhpT6oXrh-66HkenA) passward: nvg1


## Training
The training stage includes two stages:
i) Train ParsingNet. Enter parsing folder and then use main_parsingnet,py to train the model,
···
python main_parsingnet.py --writer_name parsingnet --dir_data data_path 
···
After training the ParsingNet, we use the pretrained ParsingNet to generate facial parsing,
```
python test_parsingnet.py --writer_name the_path_you_want_to_save_results
```
ii) Train FishFSRNet. Modify move the generated parsing map into the path setting of dataset_parsing.py, then train the fishfsrnet
```
python main_parsing.py --writer_name fishfsrnet --dir_data data_path
```
## Testing
The testing stage also contains two stage, we should first use the pretrained ParsingNet to estimate parsing map and then utilize the parsing map to reconstruct SR results.
```
python test_parsingnet.py --writer_name the_path_you_want_to_save_results
python test.py --writer_name fishfsrnet --dir_data data_path
```
