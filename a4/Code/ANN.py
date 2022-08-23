import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, det_curve
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

cur_dir = os.getcwd()


def Normalise(array) : 
    if(type(array) == type([])) : 
        array = np.array(array, dtype="float")
    max = np.max(array)
    min = np.min(array)
    return [(i-min)/(max-min) for i in array]

def normalise(lst):
	mn = min(lst)
	mx = max(lst)
	for i in range(len(lst)):
		lst[i] = (lst[i] - mn)/(mx - mn)
	return lst

def wind_mean(arr):
    l = len(arr)
    arr = np.array(arr)
    mean = np.multiply(np.sum(arr, axis = 0), l)
    return mean

def equalise(x_train, x_test):
    mn = 100000
    new_x_train = []
    new_x_test = []
    for j in range(len(x_train)):
        mn = min(mn, len(x_train[j]))
    for j in range(len(x_test)):
        mn = min(mn, len(x_test[j]))

    for j in range(len(x_train)):
        window = len(x_train[j]) - mn + 1
        new = []
        for k in range(len(x_train[j]) - window):
            new.append(wind_mean(x_train[j][k:k+window]))
        new_x_train.append(new)

    for j in range(len(x_test)):
        window = len(x_test[j]) - mn + 1
        new = []
        for k in range(len(x_test[j]) - window):
            new.append(wind_mean(x_test[j][k:k+window]))
        new_x_test.append(new)

    return new_x_train, new_x_test

def getSynTrainData() :
    data = np.genfromtxt(r"synthetic/train.txt", delimiter=",")
    return data

def getSynDevData() :
    data = np.genfromtxt(r"synthetic/dev.txt", delimiter=",")
    return data

def getImgTrainData() :
    train_all = []
    class_num = 1
    for class_type in ["coast","forest","highway","mountain","opencountry"] :
        X = []
        string = f"features/{class_type}" + r"/train"
        for image in os.listdir(string) : 
            file = open(string+f"/{image}","r")
            img_data = []
            for line in file: 
                temp_list = line.split(" ")
                temp_array = []
                for temp in temp_list:
                    temp_array.append(temp.split("\n")[0])
                img_data += temp_array
            temp_array2 = np.array(img_data, dtype="float").reshape(36,23)  
            temp_array3 = np.array([],dtype="float").reshape(36,0)
            for i in range(23):
                temp = Normalise(temp_array2[:,i].reshape(1,36))
                temp_array3 = np.c_[temp_array3, np.array(temp[0], dtype="float").reshape(36,1)]  
            temp_array3 = temp_array3.reshape(1,828)[0]
            temp_array4 = np.r_[temp_array3, np.array([class_num])]
            X.append(temp_array4)
        train_all += X
        class_num += 1
    return train_all

def getImgTestData() :
    test_all = []
    class_num = 1
    for class_type in ["coast","forest","highway","mountain","opencountry"] :
        X = []
        string = f"features/{class_type}" + r"/dev"
        for image in os.listdir(string) : 
            file = open(string+f"/{image}","r")
            img_data = []
            for line in file: 
                temp_list = line.split(" ")
                temp_array = []
                for temp in temp_list:
                    temp_array.append(temp.split("\n")[0])
                img_data += temp_array
            temp_array2 = np.array(img_data, dtype="float").reshape(36,23)  
            temp_array3 = np.array([],dtype="float").reshape(36,0)
            for i in range(23):
                temp = Normalise(temp_array2[:,i].reshape(1,36))
                temp_array3 = np.c_[temp_array3, np.array(temp[0], dtype="float").reshape(36,1)]  
            temp_array3 = temp_array3.reshape(1,828)[0]
            temp_array4 = np.r_[temp_array3, np.array([class_num])]
            X.append(temp_array4)
        test_all += X 
        class_num += 1
    return test_all

def getTrainData_letter_norm(l):
    ret = []
    for x in sorted(os.listdir(cur_dir + "/handwriting_data/" + l + "/train")):
        points_x = []
        points_y = []
        points_norm = []
        data_pointer = open(cur_dir + "/handwriting_data/" + l + "/train/" + x,'r')
        temp_list = data_pointer.read()
        dim = int(temp_list.split(" ")[0])
        data = [float(i) for i in temp_list.split(" ")[1:-1]]
        for i in range(dim) : 
            points_x.append(data[2*i])
            points_y.append(data[2*i + 1])
        norm_x = normalise(points_x)
        norm_y = normalise(points_y)
        for i in range(dim) : 
            points_norm.append([norm_x[i], norm_y[i]])
        ret.append(points_norm)
    return ret

def getDevData_letter_norm(l):
    ret = []
    for x in sorted(os.listdir(cur_dir + "/handwriting_data/" + l + "/dev")):
        points_x = []
        points_y = []
        points_norm = []
        data_pointer = open(cur_dir + "/handwriting_data/" + l + "/dev/" + x,'r')
        temp_list = data_pointer.read()
        dim = int(temp_list.split(" ")[0])
        data = [float(i) for i in temp_list.split(" ")[1:-1]]
        for i in range(dim) : 
            points_x.append(data[2*i])
            points_y.append(data[2*i + 1])
        norm_x = normalise(points_x)
        norm_y = normalise(points_y)
        for i in range(dim) : 
            points_norm.append([norm_x[i], norm_y[i]])
        ret.append(points_norm)
    return ret

def getTrainData_audio(l):
	ret = []
	for x in sorted(os.listdir(cur_dir + "/isolated_digits/" + l + "/train")):
		if x.endswith(".mfcc"):
			data_pointer = open(cur_dir + "/isolated_digits/" + l + "/train/" + x,'r')
			data = data_pointer.read()
			row = data.split("\n")
			dim = row[0].split(" ")
			dim = [int(i) for i in dim]
			dt = []
			for i in range(1,dim[1]+1):
				temp = row[i][1:].split(" ")
				temp = [float(j) for j in temp]
				dt.append(temp)
			ret.append(dt)
	return ret

def getDevData_audio(l):
	ret = []
	for x in sorted(os.listdir(cur_dir + "/isolated_digits/" + l + "/dev")):
		if x.endswith(".mfcc"):
			data_pointer = open(cur_dir + "/isolated_digits/" + l + "/dev/" + x,'r')
			data = data_pointer.read()
			row = data.split("\n")
			dim = row[0].split(" ")
			dim = [int(i) for i in dim]
			dt = []
			for i in range(1,dim[1]+1):
				temp = row[i][1:].split(" ")
				temp = [float(j) for j in temp]
				dt.append(temp)
			ret.append(dt)
	return ret

def PCA(x, fin):
    x = np.array(x)
    x_mean = x - np.mean(x , axis = 0)
    cov = np.cov(x_mean , rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov)

    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    eigenvector_subset = sorted_eigenvectors[:,0:fin]

    x_reduced = np.dot(eigenvector_subset.transpose() , x_mean.transpose() ).transpose()  
    return x_reduced

def LDA(input, num_classes, k): 
    num_features = input[0].shape[0]-1

    data = [np.array([]).reshape(0,num_features)] * num_classes
    for temp in input : 
        temp_class = int(temp[-1]) -1 
        temp_array = data[temp_class]
        data[temp_class] = np.r_["0,2", temp_array, temp[:-1]]
        
    means = []
    for i in range(num_classes) : 
        means.append(np.mean(data[i],axis=0))
    means = np.array(means)

    total_mean = np.mean(input[:,:-1],axis=0)
    
    within_class_scatter = np.zeros((num_features, num_features))
    for i in range(num_classes) : 
        for j in range(len(data[i])) : 
            within_class_scatter += (data[i][j] - means[i]).T @ (data[i][j] - means[i])
    
    between_class_scatter = np.zeros((num_features, num_features))
    for i in range(num_classes) : 
        between_class_scatter += data[i].shape[0] * (means[i] - total_mean).T @ (means[i] - total_mean)

    eigen_values, eigen_vectors = np.linalg.eig(np.linalg.pinv(within_class_scatter) @ between_class_scatter)
    pairs = [(np.abs(eigen_values[i]), np.abs(eigen_vectors[:,i])) for i in range(len(eigen_values))]
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

    w = np.array([]).reshape(0,num_features)
    for i in range(k) : 
        w = np.r_["0,2", w, pairs[i][1].reshape(1,num_features)]
    
    reduced_data = []
    for i in range(num_classes) : 
        reduced_data.append(data[i] @ w.T)
    
    return reduced_data 

def Plot_ROC_DET(test_class, test_score, num_classes, graph_name) :
    fpr = dict()
    tpr = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_class[:,i], test_score[:,i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i],label=f'ROC curve, class {i+1}')
        
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend()
    plt.savefig(f"ROC_{graph_name}.png")

    fpr = dict()
    fnr = dict()
    for i in range(num_classes):
        fpr[i], fnr[i], _ = det_curve(test_class[:,i], test_score[:,i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], fnr[i],label=f'DET curve, class {i+1}')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend()
    plt.savefig(f"DET_{graph_name}.png")

def plotconfusion(confusion, num_classes, graph_name):
    if(num_classes == 5):
        str = "12345"
    else:
        str = "12"
    df_cm = pd.DataFrame(confusion, index = [i for i in str], columns = [i for i in str])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f"Confusion_{graph_name}.png")

# ----------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Main Functions ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

def ANN_Syn() : 
    syn_train = getSynTrainData()
    syn_train_data = syn_train[:,:-1]
    syn_train_class = syn_train[:,-1]
    
    syn_test = getSynDevData()
    syn_test_data = syn_test[:,:-1]
    syn_test_class = syn_test[:,-1]

#------------------------------------------------ PCA ---------------------------------------------------------------
    # input = np.r_["0,2", syn_train, syn_test]
    # k = 1
    # output = PCA(input, k)

    # count = 0
    # prev_count = 0
    # syn_pca_train = np.array([]).reshape(0,k)
    # syn_pca_test = np.array([]).reshape(0,k)
    # for i in range(4) : 
    #     prev_count = count
    #     if(i < 2) :
    #         count += 1250
    #         syn_pca_train = np.r_["0,2", syn_pca_train, output[prev_count:count,:]]
    #     else : 
    #         count += 500
    #         syn_pca_test = np.r_["0,2", syn_pca_test, output[prev_count:count,:]]

#------------------------------------------------ LDA ---------------------------------------------------------------
    # input = np.r_["0,2", syn_train, syn_test]
    # k = 1
    # output = LDA(input, 2, k)

    # syn_lda_train = np.array([]).reshape(0,k)
    # syn_lda_test = np.array([]).reshape(0,k)
    # i = 0
    # for temp_array in output : 
    #     syn_lda_train = np.r_["0,2", syn_lda_train, temp_array[:1250]]
    #     syn_lda_test = np.r_["0,2", syn_lda_test, temp_array[1250:]]
    #     i += 1

#---------------------------------------------------- PLOTS -------------------------------------------------------------
    temp_test_class = syn_test_class
    syn_test_class = label_binarize(syn_test_class, classes=[1,2,3])

    # Normal 
    temp_clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=1e-3, hidden_layer_sizes=(50,10), random_state=1))
    test_score = temp_clf.fit(syn_train_data, syn_train_class).predict_proba(syn_test_data)
    pred_class = temp_clf.predict(syn_test_data)
    confusion = confusion_matrix(temp_test_class, pred_class)
    plotconfusion(confusion, 2, "ANN_Syn")
    Plot_ROC_DET(syn_test_class, test_score, 2, "ANN_Syn")

    # PCA
    # temp_clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=0.001, hidden_layer_sizes=(50,10), random_state=1))
    # test_score = temp_clf.fit(syn_pca_train, syn_train_class).predict_proba(syn_pca_test)
    # pred_class = temp_clf.predict(syn_test_data)
    # confusion = confusion_matrix(temp_test_class, pred_class)
    # plotconfusion(confusion, 2, "ANN_Syn_PCA")
    # Plot_ROC_DET(syn_test_class, test_score, 2, "ANN_Syn_PCA")

    # LDA
    # temp_clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=0.001, hidden_layer_sizes=(50,10), random_state=1))
    # test_score = temp_clf.fit(syn_lda_train, syn_train_class).predict_proba(syn_lda_test)
    # pred_class = temp_clf.predict(syn_test_data)
    # confusion = confusion_matrix(temp_test_class, pred_class)
    # plotconfusion(confusion, 2, "ANN_Syn_LDA")
    # Plot_ROC_DET(syn_test_class, test_score, 2, "ANN_Syn_LDA")




def ANN_Img() :
    img_train = getImgTrainData()
    img_train_data = []
    img_train_class = []
    train_sizes = [0] * 5
    for temp_img in img_train:
        img_train_data.append(temp_img[:-1])
        img_train_class.append(temp_img[-1])
        train_sizes[int(temp_img[-1]) - 1] += 1

    img_test = getImgTestData()
    img_test_data = []
    img_test_class = []
    test_sizes = [0] * 5
    for temp_img in img_test:
        img_test_data.append(temp_img[:-1])
        img_test_class.append(temp_img[-1])
        test_sizes[int(temp_img[-1]) - 1] += 1

#------------------------------------------------ PCA ---------------------------------------------------------------
    # input = np.r_["0,2", img_train_data, img_test_data]
    # k = 500
    # output = PCA(input, k)

    # count = 0
    # prev_count = 0
    # img_pca_train = np.array([]).reshape(0,k)
    # img_pca_test = np.array([]).reshape(0,k)
    # for i in range(10) : 
    #     prev_count = count
    #     if(i < 5) :
    #         count += train_sizes[i]
    #         img_pca_train = np.r_["0,2", img_pca_train, output[prev_count:count,:]]
    #     else : 
    #         count += test_sizes[i-5]
    #         img_pca_test = np.r_["0,2", img_pca_test, output[prev_count:count,:]]

#------------------------------------------------ LDA ---------------------------------------------------------------
    # input = np.r_["0,2", img_train, img_test]
    # k = 500
    # output = LDA(input, 5, k)

    # i = 0
    # img_lda_train = np.array([]).reshape(0,k)
    # img_lda_test = np.array([]).reshape(0,k)
    # for temp_array in output : 
    #     img_lda_train = np.r_["0,2", img_lda_train, temp_array[:train_sizes[i]]]
    #     img_lda_test = np.r_["0,2", img_lda_test, temp_array[train_sizes[i]:]]
    #     i += 1


#---------------------------------------------------- PLOTS -------------------------------------------------------------
    temp_test_class = img_test_class
    img_test_class = label_binarize(img_test_class, classes=[1,2,3,4,5])

    # Normal 
    temp_clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=0.001, hidden_layer_sizes=(50,40,10), random_state=1, max_iter=500))
    test_score = temp_clf.fit(img_train_data, img_train_class).predict_proba(img_test_data)
    pred_class = temp_clf.predict(img_test_data)
    confusion = confusion_matrix(temp_test_class, pred_class)
    plotconfusion(confusion, 5, "ANN_Image")
    Plot_ROC_DET(img_test_class, test_score, 5, "ANN_Image")

    # PCA 
    # temp_clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=0.003, hidden_layer_sizes=(50,40,50), random_state=1, max_iter=500))
    # test_score = temp_clf.fit(img_pca_train, img_train_class).predict_proba(img_pca_test)
    # pred_class = temp_clf.predict(img_test_data)
    # confusion = confusion_matrix(temp_test_class, pred_class)
    # plotconfusion(confusion, 5, "ANN_Image_PCA")
    # Plot_ROC_DET(img_test_class, test_score, 5, "ANN_Image_PCA")

    # LDA
    # temp_clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=0.001, hidden_layer_sizes=(50,10), random_state=1, max_iter=500))
    # test_score = temp_clf.fit(img_lda_train, img_train_class).predict_proba(img_lda_test)
    # pred_class = temp_clf.predict(img_test_data)
    # confusion = confusion_matrix(temp_test_class, pred_class)
    # plotconfusion(confusion, 5, "ANN_Image_LDA")
    # Plot_ROC_DET(img_test_class, test_score, 5, "ANN_Image_LDA")




def ANN_Handwritten() :
    lst = ['a', 'ai', 'bA', 'dA', 'tA']
    x_train = []
    y_train = []
    train_sizes = [0] * 5
    class_num = 1
    for l in lst:
        ret = getTrainData_letter_norm(l)
        x_train.extend(ret)
        for _ in range(len(ret)):
            y_train.append(class_num)
            train_sizes[class_num-1] += 1
        class_num += 1

    x_test = []
    y_test = []
    test_sizes = [0] * 5
    class_num = 1
    for l in lst:
        ret = getDevData_letter_norm(l)
        x_test.extend(ret)
        for _ in range(len(ret)):
            y_test.append(class_num)
            test_sizes[class_num-1] += 1
        class_num += 1

    tx_train, tx_test = equalise(x_train, x_test)
    x_train = []
    x_test = []
    for i in range(len(tx_train)):
        temp = []
        for j in range(len(tx_train[i])):
            temp.extend(tx_train[i][j])
        x_train.append(temp)

    for i in range(len(tx_test)):
        temp = []
        for j in range(len(tx_test[i])):
            temp.extend(tx_test[i][j])
        x_test.append(temp)
    
#------------------------------------------------ PCA ---------------------------------------------------------------
    # input = np.r_["0,2", x_train, x_test]
    # k = 20
    # output = PCA(input, k)

    # count = 0
    # prev_count = 0
    # handwritten_pca_train = np.array([]).reshape(0,k)
    # handwritten_pca_test = np.array([]).reshape(0,k)
    # for i in range(10) : 
    #     prev_count = count
    #     if(i < 5) :
    #         count += train_sizes[i]
    #         handwritten_pca_train = np.r_["0,2", handwritten_pca_train, output[prev_count:count,:]]
    #     else : 
    #         count += test_sizes[i-5]
    #         handwritten_pca_test = np.r_["0,2", handwritten_pca_test, output[prev_count:count,:]]

#------------------------------------------------ LDA ---------------------------------------------------------------
    # handwritten_train = np.c_[x_train, y_train]
    # handwritten_test = np.c_[x_test, y_test]
    # input = np.r_["0,2", handwritten_train, handwritten_test]
    # k = 20
    # output = LDA(input, 5, k)

    # i = 0
    # handwritten_lda_train = np.array([]).reshape(0,k)
    # handwritten_lda_test = np.array([]).reshape(0,k)
    # for temp_array in output : 
    #     handwritten_lda_train = np.r_["0,2", handwritten_lda_train, temp_array[:train_sizes[i]]]
    #     handwritten_lda_test = np.r_["0,2", handwritten_lda_test, temp_array[train_sizes[i]:]]
    #     i += 1

#---------------------------------------------------- PLOTS -------------------------------------------------------------
    temp_test_class = y_test
    y_test = label_binarize(y_test, classes=[1,2,3,4,5])

    # Normal
    clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=0.005, hidden_layer_sizes=(12,99), random_state=0, max_iter=500000))
    scores = clf.fit(x_train, y_train).predict_proba(x_test)
    pred_class = clf.predict(x_test)
    confusion = confusion_matrix(temp_test_class, pred_class)
    plotconfusion(confusion, 5, "ANN_Handwritten")
    Plot_ROC_DET(y_test, scores, 5, "ANN_Handwritten")

    # PCA
    # clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=0.005, hidden_layer_sizes=(12,99), random_state=0, max_iter=500000))
    # scores = clf.fit(handwritten_pca_train, y_train).predict_proba(handwritten_pca_test)
    # pred_class = clf.predict(x_test)
    # confusion = confusion_matrix(temp_test_class, pred_class)
    # plotconfusion(confusion, 5, "ANN_Handwritten_PCA")
    # Plot_ROC_DET(y_test, scores, 5, "ANN_Handwritten_PCA")

    # LDA
    # clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=0.005, hidden_layer_sizes=(12,99), random_state=0, max_iter=500000))
    # scores = clf.fit(handwritten_lda_train, y_train).predict_proba(handwritten_lda_test)
    # pred_class = clf.predict(x_test)
    # confusion = confusion_matrix(temp_test_class, pred_class)
    # plotconfusion(confusion, 5, "ANN_Handwritten_LDA")
    # Plot_ROC_DET(y_test, scores, 5, "ANN_Handwritten_LDA")




def ANN_Digits() :
    lst = ['1', '3', '4', '8', 'o']
    x_train = []
    y_train = []
    train_sizes = [0] * 5
    class_num = 1
    for l in lst:
        ret = getTrainData_audio(l)
        x_train.extend(ret)
        for _ in range(len(ret)):
            y_train.append(class_num)
            train_sizes[class_num-1] += 1
        class_num += 1

    x_test = []
    y_test = []
    test_sizes = [0] * 5
    class_num = 1
    for l in lst:
        ret = getDevData_audio(l)
        x_test.extend(ret)
        for _ in range(len(ret)):
            y_test.append(class_num)
            test_sizes[class_num-1] += 1
        class_num += 1

    tx_train, tx_test = equalise(x_train, x_test)
    x_train = []
    x_test = []
    for i in range(len(tx_train)):
        temp = []
        for j in range(len(tx_train[i])):
            temp.extend(tx_train[i][j])
        x_train.append(temp)

    for i in range(len(tx_test)):
        temp = []
        for j in range(len(tx_test[i])):
            temp.extend(tx_test[i][j])
        x_test.append(temp)
    
#------------------------------------------------ PCA ---------------------------------------------------------------
    # input = np.r_["0,2", x_train, x_test]
    # k = 20
    # output = PCA(input, k)

    # count = 0
    # prev_count = 0
    # digits_pca_train = np.array([]).reshape(0,k)
    # digits_pca_test = np.array([]).reshape(0,k)
    # for i in range(10) : 
    #     prev_count = count
    #     if(i < 5) :
    #         count += train_sizes[i]
    #         digits_pca_train = np.r_["0,2", digits_pca_train, output[prev_count:count,:]]
    #     else : 
    #         count += test_sizes[i-5]
    #         digits_pca_test = np.r_["0,2", digits_pca_test, output[prev_count:count,:]]

#------------------------------------------------ LDA ---------------------------------------------------------------
    # digits_train = np.c_[x_train, y_train]
    # digits_test = np.c_[x_test, y_test]
    # input = np.r_["0,2", digits_train, digits_test]
    # k = 20
    # output = LDA(input, 5, k)

    # i = 0
    # digits_lda_train = np.array([]).reshape(0,k)
    # digits_lda_test = np.array([]).reshape(0,k)
    # for temp_array in output : 
    #     digits_lda_train = np.r_["0,2", digits_lda_train, temp_array[:train_sizes[i]]]
    #     digits_lda_test = np.r_["0,2", digits_lda_test, temp_array[train_sizes[i]:]]
    #     i += 1

#---------------------------------------------------- PLOTS -------------------------------------------------------------
    temp_test_class = y_test
    y_test = label_binarize(y_test, classes=[1,2,3,4,5])

    # Normal
    # clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=0.001, hidden_layer_sizes=(20,20), random_state=0))
    # scores = clf.fit(x_train, y_train).predict_proba(x_test)
    # pred_class = clf.predict(x_test)
    # confusion = confusion_matrix(temp_test_class, pred_class)
    # plotconfusion(confusion, 5, "ANN_Digits")
    # Plot_ROC_DET(y_test, scores, 5, "ANN_Digits")

    # PCA
    # clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=0.005, hidden_layer_sizes=(20,10), random_state=0))
    # scores = clf.fit(digits_pca_train, y_train).predict_proba(digits_pca_test)
    # pred_class = clf.predict(x_test)
    # confusion = confusion_matrix(temp_test_class, pred_class)
    # plotconfusion(confusion, 5, "ANN_Digits")
    # Plot_ROC_DET(y_test, scores, 5, "ANN_Digits_PCA")

    # LDA
    # clf = OneVsRestClassifier(MLPClassifier(solver="lbfgs", alpha=0.005, hidden_layer_sizes=(20,20), random_state=0))
    # scores = clf.fit(digits_lda_train, y_train).predict_proba(digits_lda_test)
    # pred_class = clf.predict(x_test)
    # confusion = confusion_matrix(temp_test_class, pred_class)
    # plotconfusion(confusion, 5, "ANN_Digits")
    # Plot_ROC_DET(y_test, scores, 5, "ANN_Digits_LDA")
    
if __name__ == "__main__" :
    ANN_Syn()
    # ANN_Img()
    # ANN_Handwritten()
    # ANN_Digits()