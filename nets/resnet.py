import tensorflow as tf
from tensorflow.contrib import slim
from models.research.slim.nets import resnet_v1


def basenetwork(input_tensor, name='resnet50', dilation=False):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        if name == 'resnet50':
            net, end_points = resnet_v1.resnet_v1_50(input_tensor, global_pool=False)
            for key in end_points.keys():
                print key, end_points[key]
            new_end_points = {}
            new_end_points['stage0'] = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']   # 128 128 4
            new_end_points['stage1'] = end_points['resnet_v1_50/block1'] # 64 64 8
            new_end_points['stage2'] = end_points['resnet_v1_50/block2'] # 32 32 16
            new_end_points['stage3'] = end_points['final_feature']       # 16 16 32
            return net, new_end_points



if __name__ == '__main__':
    input_tensor = tf.placeholder(tf.float32, [None, 512, 512, 3])
    net, end_points = basenetwork(input_tensor)
    for key in end_points.keys():
        print(key, end_points[key])
    print('net is ', net)
