import tensorflow as tf
import numpy as np


class ABCNN:
    def __init__(self, sent, f_width, l2_reg, model_type, num_features, dim=300, num_featuremap=50, num_classes=2,
                 num_layers_cnn=2):
        """
         实现论文模型ABCNNs
        (https://arxiv.org/pdf/1512.05193.pdf)

        :param sent: 句子长度
        :param f_width: 卷积核宽度
        :param l2_reg: L2 规则
        :param model_type: 模型类型(BCNN, ABCNN1, ABCNN2, ABCNN3).
        :param num_features: The number of pre-set features(not coming from CNN) used in the output layer.
        :param dim: dimensionality 词向量维度(default: 300)
        :param num_featuremap: 卷积核个数 (default: 50)
        :param num_classes: 答案分类类型（2类）.
        :param num_layers_cnn: 卷积层的数量.
        """

        self.x1 = tf.placeholder(tf.float32, shape=[None, dim, sent], name="x1")  # 问题
        self.x2 = tf.placeholder(tf.float32, shape=[None, dim, sent], name="x2")  # 答案
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")  # 标签
        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")  # 用来分类的特征

        # 宽卷积必须零填充
        def pad_for_wide_conv(sent_seq):
            return tf.pad(sent_seq, np.array([[0, 0], [0, 0], [f_width - 1, f_width - 1], [0, 0]]), "CONSTANT",
                          name="pad_wide_conv")

        # [[0,0],[0,0],[3,3],[0,0]]

        # cosine 相似度
        def cos_sim(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

            return dot_products / (norm1 * norm2)

        def euclidean_score(v1, v2):
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
            return 1 / (1 + euclidean)

        # Attentation 矩阵
        def make_attention_mat(x1, x2):
            """
            参  数
             x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
             x2 => [batch, height, 1, width]
             [batch, width, wdith] = [batch, s, s]
            """
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
            return 1 / (1 + euclidean)

        def convolution(name_scope, input, dimi):
            with tf.name_scope(name_scope + "-conv"):
                with tf.variable_scope("conv") as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=input,
                        num_outputs=num_featuremap,
                        kernel_size=(dimi, f_width),
                        stride=1,
                        padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=True,
                        trainable=True,
                        scope=scope
                    )
                    """
                    参数解释
                    Weight: [filter_height, filter_width, in_channels, out_channels]
                    output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]
                    [batch, di, s+w-1, 1]
                    """
                    conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")  # 矩阵在dim-0上转置
                    return conv_trans

        def w_pool(variable_scope, pooling_input, attention):
            """
            参数解释
             x: [batch, di, s+w-1, 1]
             attention: [batch, s+w-1]
            """
            with tf.variable_scope(variable_scope + "-w_pool"):
                if model_type == "ABCNN2" or model_type == "ABCNN3":
                    pools = []
                    # [batch, s+w-1] => [batch, 1, s+w-1, 1]
                    attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])
                    for i in range(sent):
                        # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                        pools.append(tf.reduce_sum(pooling_input[:, :, i:i + f_width, :] *
                                                   attention[:, :, i:i + f_width, :],
                                                   axis=2,
                                                   keep_dims=True))
                    # [batch, di, s, 1]
                    w_ap = tf.concat(pools, axis=2, name="w_ap")
                else:
                    w_ap = tf.layers.average_pooling2d(
                        inputs=pooling_input,
                        # (pool_height, pool_width)
                        pool_size=(1, f_width),
                        strides=1,
                        padding="VALID",
                        name="w_ap"
                    )
                    # [batch, di, s, 1]
                return w_ap

        def all_pool(variable_scope, all_pool_input):
            with tf.variable_scope(variable_scope + "-all_pool"):
                if variable_scope.startswith("input"):
                    pool_width = sent
                    d = dim
                else:
                    pool_width = sent + f_width - 1
                    d = num_featuremap
                all_ap = tf.layers.average_pooling2d(
                    inputs=all_pool_input,
                    # (pool_height, pool_width)
                    pool_size=(1, pool_width),
                    strides=1,
                    padding="VALID",
                    name="all_ap"
                )
                # [batch, di, 1, 1]
                # [batch, di]
                all_ap_reshaped = tf.reshape(all_ap, [-1, d])
                # all_ap_reshaped = tf.squeeze(all_ap, [2, 3])
                return all_ap_reshaped

        def CNN_layer(variable_scope, input_x1, input_x2, per_dim=300):
            # x1, x2 = [batch, d, s, 1]
            with tf.variable_scope(variable_scope):
                if model_type == "ABCNN1" or model_type == "ABCNN3":
                    with tf.name_scope("att_mat"):
                        aW = tf.get_variable(name="aW",
                                             shape=(sent, per_dim),
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             regularizer=tf.contrib.lay0ers.l2_regularizer(scale=l2_reg))

                        # [batch, s, s]
                        att_mat = make_attention_mat(input_x1, input_x2)
                        # [batch, s, s] * [s,d] => [batch, s, d]
                        # matrix transpose => [batch, d, s]
                        # expand dims => [batch, d, s, 1]
                        x1_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", att_mat, aW)), -1)
                        x2_a = tf.expand_dims(tf.matrix_transpose(tf.einsum("ijk,kl->ijl", tf.matrix_transpose(att_mat),
                                                                            aW)), -1)
                        # [batch, d, s, 2]
                        input_x1 = tf.concat([input_x1, x1_a], axis=3)
                        input_x2 = tf.concat([input_x2, x2_a], axis=3)
                left_conv = convolution(name_scope="left", input=pad_for_wide_conv(input_x1), dimi=per_dim)
                right_conv = convolution(name_scope="right", input=pad_for_wide_conv(input_x2), dimi=per_dim)

                left_attention, right_attention = None, None

                if model_type == "ABCNN2" or model_type == "ABCNN3":
                    # [batch, s+w-1, s+w-1]
                    att_mat = make_attention_mat(left_conv, right_conv)
                    # [batch, s+w-1], [batch, s+w-1]
                    left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)

                left_wp = w_pool(variable_scope="left", pooling_input=left_conv, attention=left_attention)
                left_ap = all_pool(variable_scope="left", all_pool_input=left_conv)
                right_wp = w_pool(variable_scope="right", pooling_input=right_conv, attention=right_attention)
                right_ap = all_pool(variable_scope="right", all_pool_input=right_conv)

                return left_wp, left_ap, right_wp, right_ap

        x1_expanded = tf.expand_dims(self.x1, -1)
        x2_expanded = tf.expand_dims(self.x2, -1)

        LO_0 = all_pool(variable_scope="input-left", all_pool_input=x1_expanded)
        RO_0 = all_pool(variable_scope="input-right", all_pool_input=x2_expanded)

        LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope="CNN-1", input_x1=x1_expanded, input_x2=x2_expanded, per_dim=dim)
        sims = [cos_sim(LO_0, RO_0), cos_sim(LO_1, RO_1)]

        if num_layers_cnn > 1:
            _, LO_2, _, RO_2 = CNN_layer(variable_scope="CNN-2", input_x1=LI_1, input_x2=RI_1, per_dim=num_featuremap)
            self.test = LO_2
            self.test2 = RO_2
            sims.append(cos_sim(LO_2, RO_2))

        with tf.variable_scope("output-layer"):
            self.output_features = tf.concat([self.features, tf.stack(sims, axis=1)], axis=1, name="output_features")

            self.estimation = tf.contrib.layers.fully_connected(
                inputs=self.output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        self.prediction = tf.contrib.layers.softmax(self.estimation)[:, 1]

        self.cost = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.y)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="cost")

        tf.summary.scalar("cost", self.cost)
        self.merged = tf.summary.merge_all()

        print("=" * 50)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
        print("=" * 50)
