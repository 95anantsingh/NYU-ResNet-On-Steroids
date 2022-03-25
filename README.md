# **NYU-ResNet-On-Steroids**

A project based on ResNet aimed at improving ResNEt accuracy on CIFAR10 dataset while keeping the the model parameters under 5 million.

---
## **Project Stucture**



---
## **How to run**

1. To run the model at best configuration execute the following command -
    ```bash
    python resnet.py -e 100 -o adadelta -an -sc -mx -v -m project1_model
    ```

2. To resume training from best state checkpoint -
    ```bash
    python resnet.py -e 100 -o adadelta -an -sc -mx -v -m project1_model -r AA4Test
    ```
3. To save results to a file -
    ```bash
    python resnet.py -e 100 -o adadelta -an -sc -mx -v -m project1_model -r AA4Test >> ./logs/<filename>.log
    ```
    > Replace `<filename>` with your choice of file

