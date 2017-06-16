chinese code：# -*- coding:utf-8 -*-

import caffe (need the caffe path: included in eidtor
                                or caffe_root+python)

define a network(data_path, the number of data in each batch)

new a network space

network contines：

    1.Convolution
    2.Pooling
    3.InnerProduct(Full Connect)
    4.ReLU
    5.Softmax(Loss)
    
    return the params of the network
write the network into a file(train and test)

define the solver(train_net, test_net)

    1.new a SloverParameter
    2.import parameter
    3.import train net
    4.import test net
    5.测试间隔参数
    6.测试迭代次数
    7.max train times
    8.base learning rate
    9.snapshot(the number of each saving margin)
    10.snapshot_prefix (caffmodel的前缀)
    
    return SloverParameter
    
write the parameter of train and test net into
                    the prototxt of the solver

set the device when running the network

    1.set_device
    2.set_mode_gpu(cpu)
    
train and test the network
    