# **NYU-ResNet-On-Steroids**

A project aimed at improving ResNet accuracy on CIFAR10 dataset while keeping the model parameters under 5 million. The project is fully modular and device agnostic in design. New transforms can be applied from `transformers.py` and new model configurations can be loaded from `models.py` dynamically. There are multiple optional arguments to help you debug and get sense of the model.

> Best Accuracy: **94.88%**

> Parameters: **4,935,242**

---
## **Project Stucture**

|Directory / File     | Description                                 |
|---------------------|---------------------------------------------|
|`checkpoints`        | Contains the saved network states           |
|`docs`               | Project documentaion                        |
|`logs`               | Test run logs                               |
|`plots`              | Saved plots                                 |
|`models.py`          | Resnet definitions and model configuartion  |
|`resnet.py`          | main python file                            |
|`tools.py`           | Debug functions                             |
|`transformers.py`    | Data Transform definitions                  |

---
## **Run options**

To get help execute -
```bash
python resnet.py -h
```

**Optional Arguments:**
| Arguments |               | Description                  | Default
|-----------|---------------|------------------------------|-----------------
| -h       | --help        | Show this help message and exit
| -d <path>| --data_path \<path>| Dataset storage path | CIFAR10/
| -trb \<int>| --train_batch_size \<int> | Batch size for training | 128
| -vb \<int> | --val_batch_size \<int> | Batch size for validation | 256
| -tsb \<int>| --test_batch_size \<int>| Batch size for testing | 256
| -e \<int>  | --epochs \<int> | Number of epochs for training | 1
| -tsf \<str>| --test_transform \<str> | Transfom to be applied on Testing Data | TestTransform_1
| -trf \<str>| --train_transform \<str> | Transfom to be applied on Training Data | TrainTransform_1
| -r \<str>  | --resume \<str> | Load model and Resume Trainig from given filename | None
| -m \<str>  | --model \<str> | ResNet Model to be loaded from `models.py` | ResNet10_1
| -o \<str>  | --optimizer \<str> | Optimizer for the network, choices: sgd , sgdn, adagrad , adadelta , adam | sgd
| -ol \<float> | --learning_rate \<float> | Learning Rate of optimizer | 0.1
| -om \<float> | --momentum \<float> | Momentum of optimizer | 0.9
| -ow \<float> | --weight_decay \<float> | Weight Decay of optimizer | 5e-4
| -mx| --mixup        | Enable Mixup Augmentaion | false
| -sc| --schedule     | Enable Optimizer LR Decay sceduling | false
| -an| --anneal       | Enable Optimizer Annealing | false
| -v| --validate      | Enable Validation alongside Testing| false
| -s| --save          | Save best performaing network | false
| -ss| --save_state   | Save best model state dict | false
| -p| --print_summary | Print network summary | false
| -n| --no_training   | Disable Training | false
| -sp| --save_plot    | Save Error plot | false
| -pt| --plot_transforms | Save After and Before Data Transforms plots | false
| -pcm |--plot_conf_mat | Save confusion matrix plot | false
| -ofp \<str>| --output_file_prefix \<str> | Output file name prefix for saving model states and plot files | ResNet

**Usage**
> resnet.py [-h] [-d <path>] [-trb <int>] [-vb <int>] [-tsb <int>] [-e <int>] [-tsf <str>] [-trf <str>] [-r <str>] [-m <str>] [-o <str>] [-ol <float>] [-om <float>] [-ow <float>] [-mx] [-sc] [-an] [-v] [-s] [-ss] [-p] [-n] [-sp] [-pt] [-pcm] [-ofp <str>]

---
## **How to run**

1. To start training with best model configuration for 100 epochs execute the following command -
    ```bash
    python resnet.py -e 100 -o adadelta -an -sc -mx -v -m project1_model
    ```

2. To resume training  for 100 epochs from best state checkpoint -
    ```bash
    python resnet.py -e 100 -o adadelta -an -sc -mx -v -m project1_model -r AA4Test
    ```
3. To save logs to a file in logs directory -
    ```bash
    python resnet.py -e 100 -m project1_model -r AA4Test >> logs/<filename>.log
    ```
    > Replace `<filename>` with your choice of filename

</br>

---
## **Documentation**

1. Project report can be found at [docs/project_report.pdf](https://github.com/95anantsingh/NYU-ResNet-On-Steroids/tree/main/docs/project_report.pdf)
2. Logs maintained at this [Google Sheet](https://docs.google.com/spreadsheets/d/1mOcrw_yWPL7V_gtQM-UxgU_kDUZvHhMDEDV9wRH-hGY/edit?usp=sharing)

> This Project was part of graduate level Deep Learning course at New York University

