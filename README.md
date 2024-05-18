# Conditional Distribution Information Bottleneck for Distributionally Robust Sequential Recommendation

This is the code for our proposed model **CDIB**. The code is built upon [RecBole](https://github.com/RUCAIBox/RecBole).

## Requirements

The python version is `3.8.18`. Run the following code to satisfy the requirements by `pip`:

```
pip install -r requirements.txt
```

## Datasets

The `ML-100K` dataset is already located in the `\dataset` folder. To obtain the other three public datasets, you can refer to [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets). Specifically, following the steps:
- Directly download the processed atomic files (*i.e.*, Retailrocket, Amazon-Beauty, and Amazon-Sports). [Baidu Wangpan](https://pan.baidu.com/share/init?surl=p51sWMgVFbAaHQmL4aD_-g) (Password: e272), [Google Drive](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj).
- Unzip and place the datasets into the `.\dataset` folder.

## Run CDIB

Run on  `ML-100K` with the default configuration like following:

```bash
python run_recbole.py --model=CDIB --dataset=ml-100k --train_neg_sample_args=None
```

You can also add your own conguration, named `CDIB.yaml`, into the `.\recbole\properties\model` folder and run the above command. The results of the experiment will be stored in the `.\log` directory. You can review the process by examining the contents of this folder.