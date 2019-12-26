import numpy as np
import operator
import os
import inference
import tqdm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix
import codecs
import time
from functools import reduce
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import seaborn as sns


# 距离函数
def cosine_distance(x, y):
    score = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
    score = (-score+1)/2  # 距离越小越近
    return score

# 选择距离最近的K个实例
def getNeighbors(train_set, test_x, k):
    train_x, train_y = train_set
    distances = []
    # 计算测试样本与每个训练样本的距离
    for index, x in enumerate(train_x):
        dist = cosine_distance(test_x, x)
        distances.append((train_y[index], dist))
    distances.sort(key=operator.itemgetter(1))  # 按照距离进行排序  从小到大
    neighbors = distances[:k]  # 取前k个
    return neighbors

# 获取距离最近的K个实例中占比较大的类别


def get_category(neighbors):
    classVotes = {}
    for i in range(len(neighbors)):
        res = neighbors[i][0]
        if res in classVotes:
            classVotes[res] += 1
        else:
            classVotes[res] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(
        1), reverse=True)  # 对label出现的次数从大到小排序
    # print("sortedVotes=",sortedVotes)
    # print(len(sortedVotes))
    return sortedVotes[0][0]

# 训练数据集、测试数据 ，k值


def KNN(train_set, test_x, k):
    neighbors = getNeighbors(train_set, test_x, k)
    result = get_category(neighbors)
    return result


def read_data_from_npy(vec_dict):
    # 获取向量和标签
    labels = []
    x_train = None
    for vec in vec_dict['label']:
        labels.extend(vec.data.cpu().numpy().tolist())
    for vec in vec_dict['vector']:
        b = vec.data.cpu().numpy()
        if x_train is None:
            x_train = b
        else:
            x_train = np.concatenate((x_train, b), axis=0)

    y_train = np.array(labels)

    return (x_train, y_train)

# 根据保存的numpy文件生成训练数据集
def generate_train_dataset(train_npy_path, val_npy_path):
    # train_vec_dict = np.load(train_npy_path)
    val_vec_dict = np.load(val_npy_path)

    # train_vec_dict = train_vec_dict.item()  # ['vector','label']
    val_vec_dict = val_vec_dict.item()

    # x_train, y_train = read_data_from_npy(train_vec_dict)
    x_val, y_val = read_data_from_npy(val_vec_dict)

    # x_train = np.concatenate((x_train, x_val), axis=0)
    # print(x_train.shape)

    # y_train.extend(y_val)

    y_train = np.asarray(y_val)
    print(len(y_train))
    return x_val, y_train



# 预测每张图片
def predict_for_each_file(infer, test_dir, file_name):
    # 不是图片
    if not file_name.endswith('jpg'):
        return None

    with open(os.path.join(test_dir, file_name.split('.jpg')[0]+'.txt'), 'r') as f:
        line = f.readline()

    line_split = line.strip().split(',')
    if len(line_split) != 2:
        print("the content of txt is error")
        return None
    # 获取标签
    # gt_label = infer.label_id_name_dict[line_split[1].strip(' ')]

    img_path = os.path.join(test_dir, file_name)
    try:
        img = Image.open(img_path)
        img = infer.transforms(img)
        test_vector = infer._predict({"input_img": img})
        # result,pred_label = infer._inference({"input_img": img})
        # print("pred_label=",pred_label)
        y_true = line_split[1].strip(' ')
        return (y_true, test_vector)
    except Exception as e:
        print(e)
        return None


# 测试
def infer_on_dataset(train_set,test_dir, model_path,pca):
    output_dir = model_path + '_output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    infer = inference.ImageClassificationService('', model_path)

    files = os.listdir(test_dir)[:500]

    y_trues, y_preds = [], []

    error_results = []

    file_names = []
    # 测试数据降低维度
    for file_name in tqdm.tqdm(files):
        test_y = predict_for_each_file(infer, test_dir, file_name)
        if test_y:
            file_names.append(file_name)
            y_true,y_pred = test_y
            y_pred = np.squeeze(y_pred.data.cpu().numpy())

            y_preds.append(y_pred)
            y_trues.append(y_true)
        else:
            continue

    y_preds = np.asarray(y_preds)
    y_trues = np.asarray(y_trues)

    y_labels = np.array(list(map(int,y_trues)))  # 标签转化为整型数组

    # 降维
    # y_preds = use_PCA(y_preds,pca,True)

    # classes = list(set(y_labels))
    # classes.sort()
    # classnames = [infer.label_id_name_dict[str(c)] for c in classes]
    # plot_pca_scatter(y_preds,y_labels,classnames)

    y_prods = []
    
    for index,y_pred in enumerate(y_preds):
        start_time = time.time()
        #  使用knn决定类别
        result = KNN(train_set,y_pred,3)
        y_prod = infer.idx_to_class[int(result)]
        y_true = y_trues[index]
        y_prods.append(y_prod)

        end_time = time.time()
        # print("inference_time=",end_time-start_time)
        if y_true != y_prod:
            pred_label = infer.label_id_name_dict[str(y_prod)]
            true_label = infer.label_id_name_dict[str(y_true)]
            error_results.append(','.join([file_names[index],true_label,pred_label])+'\n')

    acc = accuracy_score(y_trues, y_prods)

   # 画混淆矩阵
    classes = list(set(y_trues))
    classes.sort()
    classes_name = [infer.label_id_name_dict[c] for c in classes]

    cf_matrix = confusion_matrix(y_trues,y_prods)
    inference.plotCM(cf_matrix,classes_name)

    result_file_path = os.path.join(output_dir, 'accuracy_knn.txt')

    with codecs.open(result_file_path, 'w', 'utf-8') as f:
        f.write('# predict error files\n')
        f.write('####################################\n')
        f.write('file_name, true_label, pred_label\n')
        f.writelines(error_results)
        f.write('####################################\n')
        f.write('accuracy: %s\n' % acc)
    print('accuracy result file saved as %s' % result_file_path)
    print('accuracy: %0.4f' % acc)

def use_PCA(x_train,pca,is_test=False):

    if is_test:
        X_pca=pca.transform(x_train)  # 降维后的数据
    else:
        X_pca= pca.fit_transform(x_train)
    # 输出贡献率
    # print(estimator.explained_variance_ratio_) 
    # plot_pca_scatter(X_pca,y_train,class_names)
    # print(x_train.shape)
    # print(X_pca.shape)

    return X_pca

def plot_pca_scatter(x_train,y_train,class_names):
    print(x_train.shape)
    print(y_train.shape)

    colors = ['r', 'g', 'b', 'c', 'k', 'm', 'y', 'r', 'g', 'b', 'c', 'k', 'm', 'y', 'r', 'g', 'b', 'c', 'k', 'm', 'y', 'r', 'g', 'b', 'c', 'k', 'm', 'y', 'r', 'g', 'b', 'c', 'k', 'm', 'y', 'r', 'g', 'b', 'c', 'k', 'm', 'y', 'r', 'g', 'b', 'c', 'k', 'm', 'y', 'r']

    ax = Axes3D(plt.figure())
    # for c, i, target_name in zip(colors,list(range(len(class_names))), class_names):
    #     plt.scatter(x_train[y_train==i, 0], x_train[y_train==i, 1], c=c, label=target_name)
    
    for c, i, target_name in zip(colors,list(range(len(class_names))), class_names):
        ax.scatter(x_train[y_train==i, 0], x_train[y_train==i, 1],x_train[y_train==i, 2], c=c, label=target_name)

    #设置每个坐标的取值范围
    # plt.axis([-20,20,-20,20])
    # plt.xlabel('Dimension1')
    # plt.ylabel('Dimension2')
    plt.title('data distribution')
    plt.legend()
    plt.show()
    



if __name__ == '__main__':
  
    train_npy_path = './datasets/train_epoch_33.npy'

    val_npy_path = './datasets/val_epoch_33.npy'
    

    train_x_Path = './datasets/x_train.py'
    train_y_Path = './datasets/y_train.py'

    test_dir = './../test_20'
    model_path = './cache/model_snapshots/epoch_33_96.046.pth'

    n_components = 3

    if not os.path.exists(train_x_Path):
        # 构造训练数据
        x_train,y_train = generate_train_dataset(train_npy_path,val_npy_path)
        np.save(train_x_Path,x_train)
        np.save(train_y_Path,y_train)
    else:
        x_train = np.load(train_x_Path)
        y_train = np.load(train_y_Path)
    
    # # # 降维
    pca = PCA(n_components=n_components)
    x_train = use_PCA(x_train,pca)
    print(x_train.shape)



    infer = inference.ImageClassificationService('', model_path)
    classes = list(set(y_train))
    # classes = list(map(int,classes))
    classes.sort()
    classes_name = [infer.label_id_name_dict[str(c)] for c in classes]
    plot_pca_scatter(x_train,y_train,classes_name)

    exit()

    train_set = (x_train,y_train)

    infer_on_dataset(train_set,test_dir, model_path,pca)


    # # 测试
    # test_dir = './../test_20'
    # model_path = './cache/model_snapshots/epoch_33_96.046.pth'
    # # infer_on_dataset(test_dir, model_path)

    # # 得到测试embedding
    # file_name = 'test_465.jpg'
    # # file_name = 'test_769.jpg'
    # infer = inference.ImageClassificationService('', model_path)
    # test_y = predict_for_each_file(infer, test_dir, file_name)
    # if test_y:
    #     y_true,y_pred = test_y
    # y_pred = y_pred.data.cpu().numpy()

    # print(infer.label_id_name_dict[y_true])

    # print("*"*100)

    # # 使用knn决定类别
    # result = KNN(train_set,y_pred,15)
    # print("result=",result)

    # pred_label = infer.idx_to_class[int(result)]
    # print("pred_label=",pred_label)

    # result = infer.label_id_name_dict[str(pred_label)]
    # print("result=",result)

