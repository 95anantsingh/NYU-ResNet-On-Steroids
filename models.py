
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, kernel_size, skip_kernel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=skip_kernel, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self,N:int, B:list, C:list, F:list, K:list, P:int, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = C[0]
        self.block = BasicBlock
        self.N = N                # No. of Residual Layers
        self.B = B                # No. of Residual Blocks in Residual Layer i
        self.C = C                # No. of channels in Residual Layer i
        self.F = F                # Conv. kernel size in Residual Layer i
        self.K = K                # Skip connection kernel size in Residual Layer i
        self.P = P                # Average pool kernel size
        self.layers = []          # layers container
        self.S = [2] * N          # strides for layers
        self.S[0] = 1

        # Output Liner layer input dimension
        self.outLayerInSize = C[N-1]*(32//(P*2**(N-1)))*(32//(P*2**(N-1)))

        # Print Model Config
        print("\n\nModel Config: "
            "\n-------------------------------------"
            "\nN (# Layers)\t:",self.N,
            "\nB (# Blocks)\t:",self.B,
            "\nC (# Channels)\t:",C,
            "\nF (Conv Kernel)\t:",F,
            "\nK (Skip Kernel)\t:",K,
            "\nP (Pool Kernel)\t:",P,)

        self.conv1 = nn.Conv2d(3, C[0], kernel_size=F[0], stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C[0])
        for i in range(N): 
            exec("self.layer{} = self._make_layer(self.block, self.C[{}], self.B[{}], self.F[{}], self.K[{}], self.S[{}])"\
                .format(i+1,i,i,i,i,i))
            exec("self.layers.append(self.layer{})".format(i+1))
        self.linear = nn.Linear(self.outLayerInSize, num_classes)
        

    def _make_layer(self, block, planes, num_blocks, kernel_size, skip_kernel, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, skip_kernel, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        out = F.avg_pool2d(out, self.P)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        

# N: : # Residual Layers
# Bi : # Residual blocks in Residual Layer i
# Ci : # channels in Residual Layer i
# Fi : Conv. kernel size in Residual Layer i
# Ki : Skip connection kernel size in Residual Layer i
# P  : Average pool kernel size


def project1_model():

    B=[3,3,2,3]
    C=[64,128,128,256]
    F=[3,3,3,3]
    K=[1,1,1,1]
    P=4
    N=len(B)

    return ResNet(N, B, C, F, K, P)

def ResNet10_1():

    N=4
    B=[1,1,1,1]
    C=[64,128,256,512]
    F=[3,3,3,3]
    K=[1,1,1,1]
    P=4

    return ResNet(N, B, C, F, K, P)

def ResNet10_2():

    N=3
    B=[1,1,1]
    C=[32,64,128]
    F=[3,3,3]
    K=[1,1,1]
    P=4

    return ResNet(N, B, C, F, K, P)

def ResNet24_2():

    N=3
    B=[3,3,3]
    C=[64,128,256]
    F=[3,3,3]
    K=[1,1,1]
    P=4

    return ResNet(N, B, C, F, K, P)


def ResNet34_1():

    N=4
    B=[3,4,6,3]
    C=[64,128,256,512]
    F=[3,3,3,3]
    K=[1,1,1,1]
    P=4

    return ResNet(N, B, C, F, K, P)


def ResNet16_1():

    B=[2,1,2,2]
    C=[64,128,256,256]
    F=[3,3,3,3]
    K=[1,1,1,1]
    P=4
    N=len(B)

    return ResNet(N, B, C, F, K, P)


def ResNet24_1():

    B=[3,3,2,3]
    C=[64,128,128,256]
    F=[3,3,3,3]
    K=[1,1,1,1]
    P=4
    N=len(B)

    return ResNet(N, B, C, F, K, P)


def ResNet48_1():

    B=[8,5,5,5]
    C=[64,128,128,128]
    F=[3,3,3,3]
    K=[1,1,1,1]
    P=4
    N=len(B)

    return ResNet(N, B, C, F, K, P)


def ResNet_test():

    B=[1,1,1,1]
    C=[64,128,256,512]
    F=[3,3,3,3]
    K=[1,1,1,1]
    P=4
    N=len(B)

    return ResNet(N, B, C, F, K, P)


def ResNetXYZ():

    B=[1,1,1]
    C=[64,128,128]
    F= [3,3,3]
    K= [1,1,1]
    P=4
    N=len(B)

    return ResNet(N, B, C, F, K, P)


def ResNet86():

    B= [10,8,10,14]
    C= [16,32,64,128]
    F= [3,3,3,3]
    K= [1,1,1,1]
    P= 4
    N= len(B)

    return ResNet(N, B, C, F, K, P)


def ResNet12():

    B= [2,1,1,1]
    C= [64,128,256,512]
    F= [3,3,3,3]
    K= [1,1,1,1]
    P= 4
    N= len(B)

    return ResNet(N, B, C, F, K, P)


def ResNet32():

    B= [4,4,4,3]
    C= [32,64,128,256]
    F= [3,3,3,3]
    K= [1,1,1,1]
    P= 4
    N= len(B)

    return ResNet(N, B, C, F, K, P)


def ResNet8():

    B= [1,1,1]
    C= [128,256,512]
    F= [3,3,3]
    K= [1,1,1]
    P= 4
    N= len(B)

    return ResNet(N, B, C, F, K, P)
