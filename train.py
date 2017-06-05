import sys

import numpy as np
import tensorflow as tf
from sklearn import linear_model, svm
from sklearn.externals import joblib

from ABCNN import ABCNN
from preprocess import Word2Vec, MSRP, WikiQA
from utils import build_path


def train(lr, width, l2_reg, epoch, batch_size, model_type, num_layers, data_type, word2vec, num_classes=2):
    if data_type == "WikiQA":
        train_data = WikiQA(word2vec=word2vec)
    else:
        train_data = MSRP(word2vec=word2vec)

    train_data.open_file(mode="train")

    print("=" * 50)
    print("training data size:", train_data.data_size)
    print("training max len:", train_data.max_len)
    print("=" * 50)

    model = ABCNN(sent=train_data.max_len, f_width=width, l2_reg=l2_reg, model_type=model_type,
                  num_features=train_data.num_features, num_classes=num_classes, num_layers_cnn=num_layers)

    optimizer = tf.train.AdagradOptimizer(lr, name="optimizer").minimize(model.cost)

    # Due to GTX 970 memory issues
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

    # Initialize all variables
    init = tf.global_variables_initializer()

    # model(parameters) saver
    saver = tf.train.Saver(max_to_keep=100)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_summary_writer = tf.summary.FileWriter("./tf_logs/train", sess.graph)

        sess.run(init)

        print("=" * 50)
        for e in range(1, epoch + 1):
            print("[Epoch " + str(e) + "]")

            train_data.reset_index()
            i = 0
            LR = linear_model.LogisticRegression()
            SVM = svm.LinearSVC()
            clf_features = []

            while train_data.is_available():
                i += 1
                batch_x1, batch_x2, batch_y, batch_features = train_data.next_batch(batch_size=batch_size)
                merged, _, c, features = sess.run([model.merged, optimizer, model.cost, model.output_features],
                                                  feed_dict={model.x1: batch_x1,
                                                             model.x2: batch_x2,
                                                             model.y: batch_y,
                                                             model.features: batch_features})

                clf_features.append(features)

                if i % 100 == 0:
                    print("[batch " + str(i) + "] cost:", c)
                train_summary_writer.add_summary(merged, i)

            save_path = saver.save(sess, build_path("./models/", data_type, model_type, num_layers), global_step=e)
            print("model saved as", save_path)

            clf_features = np.concatenate(clf_features)
            LR.fit(clf_features, train_data.labels)
            SVM.fit(clf_features, train_data.labels)

            LR_path = build_path("./models/", data_type, model_type, num_layers, "-" + str(e) + "-LR.pkl")
            SVM_path = build_path("./models/", data_type, model_type, num_layers, "-" + str(e) + "-SVM.pkl")
            joblib.dump(LR, LR_path)
            joblib.dump(SVM, SVM_path)

            print("LR saved as", LR_path)
            print("SVM saved as", SVM_path)

        print("training finished!")
        print("=" * 50)


if __name__ == "__main__":
    """
     参数解释
        --lr:学习率
        --ws: 窗口大小
        --l2_reg: l2_reg 
        --epoch: 训练批次
        --batch_size: batch size
        --model_type: 模型类型
        --num_layers: 卷积层数量
        --data_type: MSRP or WikiQA data
    """
    # 默认的模型超参数
    params = {
        "lr": 0.08,  # 学习率为0.08
        "ws": 4,  # 窗口大小为4
        "l2_reg": 0.0004,
        "epoch": 50,  # 迭代50次
        "batch_size": 64,  # 批大小为64
        "model_type": "BCNN",  # BaseLine Model
        "num_layers": 2,  # 2个卷积层
        "data_type": "WikiQA",
        "word2vec": Word2Vec()
    }

    print("=" * 50)
    print("***************Parameters:******************")
    for k in sorted(params.keys()):
        print(k, ":", params[k])

    # 带参数启动 启动参数如下，否则按默认参数训练
    # (training): python train.py - -lr = 0.08 - -ws = 4 - -l2_reg = 0.0004 - -epoch = 20 - -batch_size = 64 -
    # -model_type = BCNN - -num_layers = 2 - -data_type = WikiQA
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k = arg.split("=")[0][2:]
            v = arg.split("=")[1]
            params[k] = v

    train(lr=float(params["lr"]), width=int(params["ws"]), l2_reg=float(params["l2_reg"]), epoch=int(params["epoch"]),
          batch_size=int(params["batch_size"]), model_type=params["model_type"], num_layers=int(params["num_layers"]),
          data_type=params["data_type"], word2vec=params["word2vec"])
