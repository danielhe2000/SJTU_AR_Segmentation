import tensorflow as tf
import layers

def stage_1(x,training):
    """The first stage of the network
        Args:
            x: the input batch
            training: whether we are in the process of training or inference
        Returns:
            The output of the first stage
    """
    x = layers.conv(x, "conv1", 2, 3, 24, 3, training)
    x = layers.pool(x, "pool1", 2, 3)
    x = layers.down_sample_unit(x, "ds1", 24, training)
    x = layers.shuffle_net_unit(x, "sn1_1", 48, training)
    x = layers.shuffle_net_unit(x, "sn1_2", 48, training)
    stage_1_output = layers.shuffle_net_unit(x, "sn1_3", 48, training)

    return stage_1_output


def stage_2(y,training):
    """The second stage of the network
        Args:
            y: the input from stage one
            training: whether we are in the process of training or inference
        Returns:
            The output of the second stage
    """
    y = layers.down_sample_unit(y, "ds2", 48, training)
    y = layers.shuffle_net_unit(y, "sn2_1", 96, training)
    y = layers.shuffle_net_unit(y, "sn2_2", 96, training)
    y = layers.shuffle_net_unit(y, "sn2_3", 96, training)
    y = layers.shuffle_net_unit(y, "sn2_4", 96, training)
    y = layers.shuffle_net_unit(y, "sn2_5", 96, training)
    y = layers.shuffle_net_unit(y, "sn2_6", 96, training)
    stage_2_output = layers.shuffle_net_unit(y, "sn2_7", 96, training)

    return stage_2_output


def stage_3(z,training,classes):
    """The third stage of the network
        Args:
            z: the input from stage two
            training: whether we are in the process of training or inference
            classes: total number of classes
        Returns:
            The output of the third stage
    """
    z = layers.down_sample_unit(z, "ds3", 96, training)
    z = layers.shuffle_net_unit(z, "sn3_1", 192, training)
    z = layers.shuffle_net_unit(z, "sn3_2", 192, training)
    z = layers.shuffle_net_unit(z, "sn3_3", 192, training)
    z = layers.conv(z, "conv2", 1, 192, classes, 1, training)
    stage_3_output = layers.up_sample_unit(z,"us1",2,classes)

    return stage_3_output



def clock_shufflenet(x,training, classes, update_3, update_2, last_frame_stage2,last_frame_stage3):
    """The whole network
        Args:
            x: the input batch
            training: whether we are in the process of training or inference
            classes: total number of classes
            update_3: True: update stage 3; False: use stage 3 output from last frame directly. When training always true
            update_2: True: update stage 2; False: use stage 2 output from last frame directly. When training always true
            last_frame_stage3: Stage three from last frame
            last_frame_stage2: Stafe two from last frame
        Returns:
            The output of the third stage
    """

    stage1 = stage_1(x,training)
    stage2 = tf.case([(update_2,stage_2(stage1,training))],default=last_frame_stage2)
    stage3 = tf.case([(update_3,stage_3(stage2,training,classes))],default=last_frame_stage3)

    ########DECODER#######
    stage1_conv = layers.conv(stage1,"conv4",1,48,classes,1,training)
    stage2_conv = layers.conv(stage2,"conv3",1,96,classes,1,training)

    stage_2_and_3 = tf.add(stage2_conv,stage3)
    stage_2_and_3 = layers.up_sample_unit(stage_2_and_3,"us2",2,classes)

    stage_123 = tf.add(stage_2_and_3,stage1_conv)

    final_output = layers.up_sample_unit(stage_123,"us3",8,classes)

    return final_output, stage2, stage3



