import math
import random
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from sklearn.svm import SVC 
from sklearn import metrics


cur_dir = os.getcwd() 

def normalise(lst):
    mn = min(lst)
    mx = max(lst)
    for i in range(len(lst)):
        lst[i] = (lst[i] - mn)/(mx - mn)
    return lst

def getSynTrainData() :
    data = np.genfromtxt(r"synthetic/train.txt", delimiter=",")
    return data

def getSynDevData() :
    data = np.genfromtxt(r"synthetic/dev.txt", delimiter=",")
    return data

def getImgTrainData() :
    train_all = []
    for class_type in ["coast","forest","highway","mountain","opencountry"] : 
        X = [np.array([]).reshape(0,23)] * 36
        string = f"features/{class_type}" + r"/train"
        for image in os.listdir(string) : 
            file = open(string+f"/{image}","r")
            ind = 0
            for line in file: 
                temp_list = line.split(" ")
                temp_array = []
                for temp in temp_list:
                    temp_array.append(temp.split("\n")[0])
                temp_array = np.array(temp_array, dtype="float")
                X[ind] = np.r_["0,2", X[ind], (temp_array)]
                ind += 1
        train_all.append(X)  
    return train_all

def getImgTestData() :
    test_all = []
    for class_type in ["coast","forest","highway","mountain","opencountry"] : 
        X = []
        string = f"features/{class_type}" + r"/dev"
        for image in os.listdir(string) : 
            Y = []
            file = open(string+f"/{image}","r")
            for line in file: 
                temp_list = line.split(" ")
                temp_array = []
                for temp in temp_list:
                    temp_array.append(temp.split("\n")[0])
                temp_array = np.array(temp_array, dtype="float")
                Y.append(temp_array)
            X.append(Y)
        test_all.append(X)  
    return test_all

def getTrainData_letter_norm(l):
    ret = []
    for x in sorted(os.listdir(cur_dir + "/TeluguLetter/" + l + "/train")):
        points_x = []
        points_y = []
        points_norm = []
        data_pointer = open(cur_dir + "/TeluguLetter/" + l + "/train/" + x,'r')
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
    for x in sorted(os.listdir(cur_dir + "/TeluguLetter/" + l + "/dev")):
        points_x = []
        points_y = []
        points_norm = []
        data_pointer = open(cur_dir + "/TeluguLetter/" + l + "/dev/" + x,'r')
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
    for x in os.listdir(cur_dir + "/Audio/" + l + "/train"):
        if x.endswith(".mfcc"):
            data_pointer = open(cur_dir + "/Audio/" + l + "/train/" + x,'r')
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
    for x in os.listdir(cur_dir + "/Audio/" + l + "/dev"):
        
        if x.endswith(".mfcc"):
            data_pointer = open(cur_dir + "/Audio/" + l + "/dev/" + x,'r')
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

def min_max(arr):
    arr = np.array(arr)
    mn = arr.min()
    mx = arr.max()
    arr = (arr - mn)/(mx - mn)
    return arr, mx, mn

def genPhi(x,degree):
    n = len(x)
    phi = []
    count = 0
    cnt = 0
    for x_i in x:
        cnt += 1
        num = len(x_i)
        dp = []
        dp.append(1)
        if degree == 0:
            phi.append(dp)
            count = len(dp)
            continue
        for xx in x_i:
            dp.append(xx)
        if degree == 1:
            phi.append(dp)
            count = len(dp)
            continue
        for xx in x_i:
            for yy in x_i:
                dp.append(xx*yy)
        if degree == 2:
            phi.append(dp)
            count = len(dp)
            continue    

    phi = np.array(phi)
    return phi, count


def genPhi_all(x,degree):
    n = len(x)
    # print(x)
    phi = []
    count = 0
    for x_i in x:
        num = len(x_i)
        dp = []
        dp.append(1)
        if degree == 0:
            phi.append(dp)
            continue
        for xx in x_i:
            dp.append(xx)
        if degree == 1:
            phi.append(dp)
            continue
        for xx in x_i:
            for yy in x_i:
                phi.append(xx*yy)
        if degree == 2:
            phi.append(dp)
            continue  
    # print(phi)         
    count = len(phi[0])
    phi = np.array(phi)
    return phi, count

def sigmoid(a):
    if a > 100 :
        return 1 - 1e-50
    elif a < -100:
        return 1e-50
    x = math.exp(-a)
    return 1/(1 + x)

def cost_function(x, y, w, degree):
    cost = 0
    phi, cnt = genPhi(x,degree)
    for i in range(phi.shape[0]):
        phi_j = phi[i].reshape(1, phi[i].shape[0])
        h_i = sigmoid(np.dot(phi_j, w))
        cost += y[i] * np.log(h_i) + (1 - y[i]) * np.log(1 - h_i)
    cost = -cost
    return cost/x.shape[0]

def getW(x, y, degree, learn_rate):
    x = np.array(x)
    y = np.array(y)
    phi, count = genPhi(x,degree)
    w = np.zeros((count, 1))
    w[0] = 1
    cost_all = []

    for i in range(200):
        summ = np.zeros((phi.shape[1],1))
        for j in range(x.shape[0]):
            phi_j = phi[j].reshape(1, phi[j].shape[0])
            summ = np.add(summ, (sigmoid(np.dot(phi_j, w)) - y[j]) * phi_j.transpose())
        w = w - learn_rate * summ
        pp = cost_function(x,y,w,degree)
        cost_all.append(pp)

    return w, cost_all

def test_syn(x,w, degree):
    x = np.array(x)
    # y = np.array(y)
    pred = [] 
    prob = []   
    phi, count = genPhi(x,degree)
    for i in range(phi.shape[0]):
        phi_j = phi[i].reshape(1, phi[i].shape[0])
        h_i = sigmoid(np.dot(phi_j, w))
        prob.append(h_i)
        if h_i >= 0.5 :
            pred.append(1)
        else:
            pred.append(0)

    # print(prob)

    return pred,prob

def PCA(x, fin):
    # print(x)
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

def rocdet(st_roc, st_det ,y_test, decision_func):
    fpr, tpr, _ = metrics.roc_curve(y_test,  decision_func)
    roc_auc = metrics.auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr,tpr,label = "Area under the curve = " + str(roc_auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = "lower right")
    plt.savefig(st_roc + ".png")

    fpr, fnr, _ = metrics.det_curve(y_test,  decision_func)
    plt.clf()
    plt.plot(fpr,fnr)
    plt.xlabel("False Alarm Rate")
    plt.ylabel("Missed Detection Rate")
    # plt.legend(loc = "lower right")
    plt.savefig(st_det + ".png")

def LogisticReg_Syn() :
    syn_input = getSynTrainData()
    syn_data = np.array(syn_input[:,:2], dtype="float")
    syn_data1 = np.array(syn_data[:1250,:])
    syn_data2 = np.array(syn_data[1250:,:])
    syn_class = np.array(syn_input[:,2], dtype="int")

    syn_test_input = getSynDevData()
    syn_test1 = np.array(syn_test_input[:500,:2], dtype="float")
    syn_test2 = np.array(syn_test_input[500:,:2], dtype="float")
    degree = 1
    print("Input done")
    syn_class -= 1
    learn_rate = 0.000001
    w, cost_all = getW(syn_data, syn_class, degree, learn_rate)
    print("w done")
    pred, prob = test_syn(syn_test_input[:1000,:2], w, degree)
    true = 0
    false = 0
    test_class = []
    for i in range(500):
        test_class.append(0)
    for i in range(500):
        test_class.append(1)

    model_true_x = []
    model_true_y = []
    model_false_x = []
    model_false_y = []
    for i in range(len(syn_test_input)):
        if pred[i] == 0 and i >= 500:
            false += 1
        elif pred[i] == 1 and i < 500:
            false += 1
        else:
            true += 1

        if pred[i] == 0:
            model_false_x.append(syn_test_input[i][0])
            model_false_y.append(syn_test_input[i][1])
        else:
            model_true_x.append(syn_test_input[i][0])
            model_true_y.append(syn_test_input[i][1])

    x_axis = []
    for i in range(len(cost_all)):
        x_axis.append(i+1)


    plt.clf()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Precdiction of logistic regression model')
    plt.scatter(model_true_x, model_true_y, color = 'blue')
    plt.scatter(model_false_x, model_false_y, color = 'red')
    plt.savefig('scatter_svm_syn.png')
    rocdet("log_ROC_syn", "log_DET_syn", test_class, prob)
    print(true, false)

def oneVSall(x_train, y_train, x_test, y_test, degree, num_classes):
    true = 0
    false = 0
    learn_rate = 0.0000001
    new_x = []
    new_x.extend(x_train)
    new_x.extend(x_test)
    new_x = PCA(new_x, 20)
    l = len(x_train)
    x_train = new_x[:l]
    x_test = new_x[l:]
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    plt.xlabel("False Alarm Rate")
    plt.ylabel("Missed Detection Rate")
    for i in range(num_classes):
        cl_true = 0
        cl_false = 0
        new_y = []
        for val in y_train:
            if val == i:
                new_y.append(0)
            else:
                new_y.append(1)
        w, cost_all = getW(x_train, new_y, degree, learn_rate)
        pred, prob = test_syn(x_test, w, degree)
        print(prob, pred)
        for j in range(len(y_test)):
            if y_test[j] == i:
                # print(pred[j])
                if pred[j] == 0:
                    cl_true += 1
                else:
                    cl_false += 1
        print("Class " + str(i+1) + ": true = " + str(cl_true) + ", false = " + str(cl_false))
        true += cl_true
        false += cl_false
        dec_temp = []
        y_temp = []
        for j in range(len(x_test)):
            dec_temp.append(prob[j])
            if y_test[j] == i:
                y_temp.append(0)
            else:
                y_temp.append(1)
        # print(len(y_temp), len(dec_temp[0]))
        # fpr, tpr, _ = metrics.det_curve(y_temp, dec_temp)
        # roc_auc = metrics.auc(fpr, tpr)
        # plt.plot(fpr, tpr, label = "(Class " + str(i+1) + ")")

    print(true, false)
        

    # plt.legend(loc = 'best')
    # # rocdet("ROC_svm_img_linear", "DET_svm_img_linear", y_test, decision_func)
    # plt.savefig('log_DET_audio.png')



def LogisticReg_Img() :

    img_train = getImgTrainData()
    x_train = []
    y_train = []
    for i in range(5):
        num_data = len(img_train[i][0])
        for j in range(num_data):
            y_train.append(i)
            temp = []
            for k in range(36):
                for l in range(23):
                    temp.append(img_train[i][k][j][l])
            x_train.append(temp)
            # print(temp, "aaa")

    img_test = getImgTestData()
    x_test = []
    y_test = []
    for i in range(5):
        for j in range(len(img_test[i])):
            temp = []
            for k in range(36):
                for l in range(23):
                    temp.append(img_test[i][j][k][l])
            x_test.append(temp)
            y_test.append(i)

    # x_train, mx, mn = min_max(x_train)
    # x_test = (x_test - mn)/(mx - mn)
    # # print(x_test, x_train)
    degree = 1
    print("Input done")
    oneVSall(x_train, y_train, x_test, y_test, degree, 5)
    plt.clf()

def svm_syn():
    syn_input = getSynTrainData()
    syn_data = np.array(syn_input[:,:2], dtype="float")
    syn_data1 = np.array(syn_data[:1250,:])
    syn_data2 = np.array(syn_data[1250:,:])
    syn_class = np.array(syn_input[:,2], dtype="int")

    syn_test_input = getSynDevData()
    syn_test1 = np.array(syn_test_input[:500,:2], dtype="float")
    syn_test2 = np.array(syn_test_input[500:,:2], dtype="float")

    # degree = 2
    print("Input done")
    y_train = []
    for _ in range(1250):
        y_train.append(0)
    for _ in range(1250):
        y_train.append(1)

    y_test = []
    for _ in range(500):
        y_test.append(0)
    for _ in range(500):
        y_test.append(1)

    clf = SVC(kernel='rbf', C = 2000)
    clf.fit(syn_data, y_train)
    true = 0
    false = 0
    syn_test = []
    model_true_x = []
    model_true_y = []
    model_false_x = []
    model_false_y = []
    cnt = 0
    for p in syn_test1:
        syn_test.append(p)
        cl = clf.predict(p.reshape(1,2))
        if cl == 0:
            true+=1
        else:
            false+=1
        if cl == 0:
            model_false_x.append(syn_test_input[cnt][0])
            model_false_y.append(syn_test_input[cnt][1])
        else:
            model_true_x.append(syn_test_input[cnt][0])
            model_true_y.append(syn_test_input[cnt][1])
        cnt += 1

    for p in syn_test2:
        syn_test.append(p)
        cl = clf.predict(p.reshape(1,2))
        if cl == 0:
            false+=1
        else:
            true+=1

        if cl == 0:
            model_false_x.append(syn_test_input[cnt][0])
            model_false_y.append(syn_test_input[cnt][1])
        else:
            model_true_x.append(syn_test_input[cnt][0])
            model_true_y.append(syn_test_input[cnt][1])
        cnt += 1

    print(true, false)
    decision_func = clf.decision_function(syn_test)
    # print(len(y_test), len(decision_func))
    plt.clf()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Precdiction of SVM model(rbf)')
    plt.scatter(model_true_x, model_true_y, color = 'blue')
    plt.scatter(model_false_x, model_false_y, color = 'red')
    plt.savefig('scatter_SVM_syn_rbf.png')
    rocdet("ROC_svm_syn_rbf", "DET_svm_syn_rbf", y_test, decision_func)

def plotconfusion(confusion):
    df_cm = pd.DataFrame(confusion, index = [i for i in "12345"], columns = [i for i in "12345"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("letter_confusion.png")

def svm_Img():
    img_train = getImgTrainData()
    x_train = []
    y_train = []
    for i in range(5):
        num_data = len(img_train[i][0])
        for j in range(num_data):
            y_train.append(i)
            temp = []
            for k in range(36):
                for l in range(23):
                    temp.append(img_train[i][k][j][l])
            x_train.append(temp)
            # print(temp, "aaa")

    img_test = getImgTestData()
    x_test = []
    y_test = []
    for i in range(5):
        for j in range(len(img_test[i])):
            temp = []
            for k in range(36):
                for l in range(23):
                    temp.append(img_test[i][j][k][l])
            x_test.append(temp)
            y_test.append(i)

    
    size = 36*23
    # new_x = []
    # new_x.extend(x_train)
    # new_x.extend(x_test)
    # new_x = PCA(new_x, size)
    # l = len(x_train)
    # x_train = new_x[:l]
    # x_test = new_x[l:]
    clf = SVC(kernel='rbf', C = 500000)
    # clf = SVC(kernel = 'rbf', C = 20)
    # x_train = np.array(x_train)
    print(len(x_train))
    clf.fit(x_train, y_train)
    true = 0
    false = 0
    confusion = np.zeros((5,5))
    for i in range(len(x_test)):
        cl = clf.predict(np.array(x_test[i]).reshape(1,size))
        # print(cl, y_train)
        confusion[y_test[i]][cl] += 1
        if cl == y_test[i]:
            true+=1
        else:
            false+=1
    print(true, false)
    plotconfusion(confusion)
    decision_func = clf.decision_function(x_test)

    plt.clf()
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    plt.xlabel("False Alarm Rate")
    plt.ylabel("Missed Detection Rate")
    for i in range(5):
        dec_temp = []
        y_temp = []
        for j in range(len(x_test)):
            dec_temp.append(decision_func[j][i])
            if y_test[j] == i:
                y_temp.append(1)
            else:
                y_temp.append(0)
        # print(len(y_temp), len(dec_temp[0]))
        fpr, tpr, _ = metrics.det_curve(y_temp, dec_temp)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label = '(Class ' + str(i+1) + ')')

    plt.legend(loc = 'best')
    # rocdet("ROC_svm_img_linear", "DET_svm_img_linear", y_test, decision_func)
    plt.savefig('DET_svm_img_pca.png')

def wind_mean(arr):
    l = len(arr)
    arr = np.array(arr)
    mean = np.multiply(np.sum(arr, axis = 0), l)
    return mean

def equalise(x_train, x_test):
    mn = 100000
    new_x_train = []
    new_x_test = []
    # x_train = np.array(x_train)
    # x_test = np.array(x_test)
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


def svm_timeseries_telugu():
    print('TELUGU LETTER-HANDWRITTEN')
    lst = ['a', 'ai', 'bA', 'dA', 'tA']
    x_train = []
    y_train = []
    cnt = 0
    for l in lst:
        ret = getTrainData_letter_norm(l)
        x_train.extend(ret)
        for _ in range(len(ret)):
            y_train.append(cnt)
        cnt += 1

    x_test = []
    y_test = []
    cnt = 0
    for l in lst:
        ret = getDevData_letter_norm(l)
        x_test.extend(ret)
        for _ in range(len(ret)):
            y_test.append(cnt)
        cnt += 1

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

    clf = SVC(kernel='rbf', C = 200)
    clf.fit(x_train, y_train)
    true = 0
    false = 0
    size = len(x_train[0])
    confusion = np.zeros((5,5))
    for i in range(len(x_test)):
        cl = clf.predict(np.array(x_test[i]).reshape(1,size))
        # print(cl, y_train)
        confusion[y_test[i]][cl] += 1
        if cl == y_test[i]:
            true+=1
        else:
            false+=1
    print(true, false)
    plotconfusion(confusion)
    decision_func = clf.decision_function(x_test)

    plt.clf()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.xlabel("False Alarm Rate")
    # plt.ylabel("Missed Detection Rate")
    for i in range(5):
        dec_temp = []
        y_temp = []
        for j in range(len(x_test)):
            dec_temp.append(decision_func[j][i])
            if y_test[j] == i:
                y_temp.append(1)
            else:
                y_temp.append(0)
        # print(len(y_temp), len(dec_temp[0]))
        fpr, tpr, _ = metrics.roc_curve(y_temp, dec_temp)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label = 'AUC (Class ' + str(i+1) + ') = ' + str(roc_auc))

    plt.legend(loc = 'best')
    # rocdet("ROC_svm_img_linear", "DET_svm_img_linear", y_test, decision_func)
    plt.savefig('ROC_svm_telugu.png')

def svm_timeseries_digit():
    print('DIGIT-AUDIO')
    lst = ['1','3','4','8','o']
    x_train = []
    y_train = []
    cnt = 0
    for l in lst:
        ret = getTrainData_audio(l)
        x_train.extend(ret)
        for _ in range(len(ret)):
            y_train.append(cnt)
        cnt += 1

    x_test = []
    y_test = []
    cnt = 0
    for l in lst:
        ret = getDevData_audio(l)
        x_test.extend(ret)
        for _ in range(len(ret)):
            y_test.append(cnt)
        cnt += 1

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

    clf = SVC(kernel='rbf', C = 220)
    clf.fit(x_train, y_train)
    true = 0
    false = 0
    size = len(x_train[0])
    confusion = np.zeros((5,5))
    for i in range(len(x_test)):
        cl = clf.predict(np.array(x_test[i]).reshape(1,size))
        # print(cl, y_train)
        confusion[y_test[i]][cl] += 1
        if cl == y_test[i]:
            true+=1
        else:
            false+=1
    print(true, false)
    plotconfusion(confusion)
    decision_func = clf.decision_function(x_test)

    plt.clf()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.xlabel("False Alarm Rate")
    # plt.ylabel("Missed Detection Rate")
    for i in range(5):
        dec_temp = []
        y_temp = []
        for j in range(len(x_test)):
            dec_temp.append(decision_func[j][i])
            if y_test[j] == i:
                y_temp.append(1)
            else:
                y_temp.append(0)
        # print(len(y_temp), len(dec_temp[0]))
        fpr, tpr, _ = metrics.roc_curve(y_temp, dec_temp)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label = 'AUC (Class ' + str(i+1) + ') = ' + str(roc_auc))

    plt.legend(loc = 'best')
    # rocdet("ROC_svm_img_linear", "DET_svm_img_linear", y_test, decision_func)
    plt.savefig('ROC_svm_audio.png')

def log_timeseries_telugu():
    print('TELUGU LETTER-HANDWRITTEN')
    lst = ['a', 'ai', 'bA', 'dA', 'tA']
    x_train = []
    y_train = []
    cnt = 0
    for l in lst:
        ret = getTrainData_letter_norm(l)
        x_train.extend(ret)
        for _ in range(len(ret)):
            y_train.append(cnt)
        cnt += 1

    x_test = []
    y_test = []
    cnt = 0
    for l in lst:
        ret = getDevData_letter_norm(l)
        x_test.extend(ret)
        for _ in range(len(ret)):
            y_test.append(cnt)
        cnt += 1

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

    oneVSall(x_train, y_train, x_test, y_test,1 , 5)

def log_timeseries_digit():
    print('DIGIT-AUDIO')
    lst = ['1','3','4','8','o']
    x_train = []
    y_train = []
    cnt = 0
    for l in lst:
        ret = getTrainData_audio(l)
        x_train.extend(ret)
        for _ in range(len(ret)):
            y_train.append(cnt)
        cnt += 1

    x_test = []
    y_test = []
    cnt = 0
    for l in lst:
        ret = getDevData_audio(l)
        x_test.extend(ret)
        for _ in range(len(ret)):
            y_test.append(cnt)
        cnt += 1

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

    oneVSall(x_train, y_train, x_test, y_test,1 , 5)
    

if __name__ == "__main__" :
    LogisticReg_Syn()
    LogisticReg_Img()
    svm_syn()
    svm_Img()
    svm_timeseries_telugu()
    svm_timeseries_digit()
    log_timeseries_telugu()
    log_timeseries_digit()
    