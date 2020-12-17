### A Demo of Applying TurboTransformers in Translation Task

### Prerequisites
[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py "OpenNMT-py")

[Tencent/TurboTransformers](https://github.com/Tencent/TurboTransformers "TurboTransformers")

#### Usage
```
mkdir -p model/3_18
# download a pretrained translation model in that dir
# https://pan.baidu.com/s/1iWAtd4gYt7l2f4rXZoucxQ
# Password Code : su5x
https://pan.baidu.com/s/1iWAtd4gYt7l2f4rXZoucxQ
pip install -r requirements.txt
python ./test_predict_local.py [--use_gpu]
```

#### Known Issues
OpenNMT-py==1.1.0 will throw errors on some test cases. Upgrade it tp 1.2.0.


