# -*- coding:utf-8 -*-
import caffe
from caffe import layers as L, params as P, proto

def lenet(lmdb, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()

    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss =  L.SoftmaxWithLoss(n.score, n.label)

    return n.to_proto()

with open('mnist/lenet_auto_train.prototxt', 'w') as f:
    f.write(str(lenet('/home/liuchenfeng/caffe/examples/mnist/mnist_train_lmdb', 64)))

with open('mnist/lenet_auto_test.prototxt', 'w') as f:
    f.write(str(lenet('/home/liuchenfeng/caffe/examples/mnist/mnist_test_lmdb', 100)))

def gen_solver(train_net,test_net):
    s=proto.caffe_pb2.SolverParameter()
    s.train_net =train_net
    s.test_net.append(test_net)
    s.test_interval = 938    #60000/64，测试间隔参数：训练完一次所有的图片，进行一次测试
    s.test_iter.append(100)  #10000/100 测试迭代次数，需要迭代100次，才完成一次所有数据的测试
    s.max_iter = 9380       #10 epochs , 938*10，最大训练次数
    s.base_lr = 0.01    #基础学习率
    s.momentum = 0.9    #动量
    s.weight_decay = 5e-4  #权值衰减项
    s.lr_policy = 'step'   #学习率变化规则
    s.stepsize=3000         #学习率变化频率
    s.gamma = 0.1          #学习率变化指数
    s.display = 20         #屏幕显示间隔
    s.snapshot = 938       #保存caffemodel的间隔
    s.snapshot_prefix = 'mnist/lenet/model'   #caffemodel前缀
    s.type ='SGD'         #优化算法
    s.solver_mode = proto.caffe_pb2.SolverParameter.GPU    #加速
    #写入solver.prototxt
    return s
with open('mnist/lenet_solver.prototxt', 'w') as f:
    f.write(str(gen_solver('mnist/lenet_auto_train.prototxt',
                           'mnist/lenet_auto_test.prototxt')))
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('mnist/lenet_solver.prototxt')
solver.net.copy_from('mnist/lenet/model_iter_2814.caffemodel')
solver.solve()