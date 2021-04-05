from os import listdir, getcwd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
import time
from joblib import dump, load
  

def xor(a, b):
    return a != b
      
def get_filenames(emotion):
    path = '%s\\Baza CK+\\%s' % (getcwd(), emotion)
    filenames = [f for f in listdir(path) if f.endswith('.png')]
    return filenames

def desc_LNEP(image):
    size = image.shape
    if size[2]: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    I4 = np.zeros((image.shape))
    LNEP = np.zeros((image.shape))
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            I4[i-1,j-1] = not xor(image[i,j-1] >= image[i-1,j-1], image[i-1,j] >= image[i-1,j-1])
            I4[i-1,j]   = not xor(image[i-1,j-1]>=image[i-1,j], image[i-1,j+1]>=image[i-1,j])
            I4[i-1,j+1] = not xor(image[i-1,j]>=image[i-1,j+1], image[i,j+1]>=image[i-1,j+1])
            I4[i,j-1]   = not xor(image[i-1,j-1]>=image[i,j-1], image[i+1,j-1]>=image[i,j-1])
            I4[i,j+1]   = not xor(image[i-1,j+1]>=image[i,j+1], image[i+1,j+1]>=image[i,j+1])
            I4[i+1,j-1] = not xor(image[i,j-1]>=image[i+1,j-1], image[i+1,j]>=image[i+1,j-1])
            I4[i+1,j]   = not xor(image[i+1,j-1]>=image[i+1,j], image[i+1,j+1]>=image[i+1,j])
            I4[i+1,j+1] = not xor(image[i+1,j]>=image[i+1,j+1], image[i,j+1]>=image[i+1,j+1])
            LNEP[i,j]   = int(I4[i-1,j+1]*(2**7)+I4[i-1,j]*(2**6)+I4[i-1,j-1]*(2**5)+I4[i,j-1]*(2**4)+
                              I4[i+1,j-1]*(2**3)+I4[i+1,j]*(2**2)+I4[i+1,j+1]*2+I4[i,j+1])
    return np.uint8(LNEP)

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):

    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     
    val_ar.append(get_pixel(img, center, x, y+1))        
    val_ar.append(get_pixel(img, center, x+1, y+1))     
    val_ar.append(get_pixel(img, center, x+1, y))       
    val_ar.append(get_pixel(img, center, x+1, y-1))     
    val_ar.append(get_pixel(img, center, x, y-1))       
    val_ar.append(get_pixel(img, center, x-1, y-1))     
    val_ar.append(get_pixel(img, center, x-1, y))       
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    
    
def desc_LBP(image):
    height, width, channel = image.shape
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             image_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)                            
    return cv2.cvtColor(image_lbp, cv2.COLOR_BGR2GRAY)

def show(image):
    image = cv2.resize(image, (image.shape[0]*8, image.shape[1]*8))
    cv2.imshow('out', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows
        
def demo(n, emotion):
    filenames = get_filenames(emotion)
    path = '%s\\Baza CK+\\%s\\%s' % (getcwd(), emotion, filenames[n])
    image = cv2.imread(path)
    lnep = desc_LNEP(image)
    lbp = desc_LBP(image)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Zdjecie oryginalne')
    
    plt.subplot(1, 3, 2)
    plt.imshow(lnep, cmap='gray')
    plt.axis('off')
    plt.title('LNEP')
    
    plt.subplot(1, 3, 3)
    plt.imshow(lbp, cmap='gray')
    plt.axis('off')
    plt.title('LBP')
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(lbp.ravel(), 256, [0, 256])
    plt.title('LBP')
    
    plt.subplot(1, 2, 2)
    plt.hist(lnep.ravel(), 256, [0, 256])
    plt.title('LNEP')
    
    plt.figure()
    #plt.subplot(1, 3, 3)
    hist_lnep = cv2.calcHist([lnep], [0], None, [256], [0, 256])
    hist_lbp = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    hist_sum = np.concatenate((np.squeeze(hist_lbp), np.squeeze(hist_lnep)))
    bins = [i for i in range(len(hist_sum))]
    plt.bar(bins, hist_sum)
    plt.title('Suma')

def get_hist(image):
    lnep = desc_LNEP(image)
    lbp = desc_LBP(image)
    hist_lnep = cv2.calcHist([lnep], [0], None, [128], [0, 256])
    hist_lbp = cv2.calcHist([lbp], [0], None, [128], [0, 256])
    hist_sum = np.concatenate((np.squeeze(hist_lbp), np.squeeze(hist_lnep)))    
    return hist_sum

def get_data():
    X = []  # histogramy
    Y = []  # etykiety np.'anger', 'happiness'
    path = '%s\\Baza CK+' % (getcwd())
    emotions = [directories for directories in listdir(path)]
    for emotion in emotions:
        print(emotion)
        filenames = get_filenames(emotion)
        for filename in filenames:
            full_path = '%s\\%s\\%s' % (path, emotion, filename)
            image = cv2.imread(full_path)
            hist_sum = get_hist(image)
            X.append(hist_sum)
            Y.append(emotion)
            
    #X_r = select_features(X, Y)
    X_norm = preprocessing.normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, test_size=0.33, random_state=42)
    print('Data loaded')
    return X_train, X_test, y_train, y_test

def select_features(X, y):
	fs = SelectKBest(score_func=chi2, k=200)
	fs.fit(X, y)
	X_reduced = fs.transform(X)
	return X_reduced


def train(X_train, X_test, y_train, y_test):
    parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1, 10, 100, 1000], 'gamma':[20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 1000]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, y_train)
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
    path = '%s\\Baza CK+' % (getcwd())
    emotions = [directories for directories in listdir(path)]
    for title, normalize in titles_options:
        
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                     display_labels=emotions,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
    
        #print(title)
        #print(disp.confusion_matrix)
    
    plt.show()
    return clf

def get_model():
    tic = time.perf_counter()
    X_train, X_test, Y_train, Y_test = get_data()
    
    print(str(X_train.shape) + 'X train')
    print(str(X_test.shape) + 'X test')
    toc1 = time.perf_counter()
    clf = train(X_train, X_test, Y_train, Y_test)
    y_predicted = clf.predict(X_test)
    print('C: %d, gamma: %d' % (clf.best_estimator_.C, clf.best_estimator_.gamma))
    print(accuracy_score(Y_test, y_predicted))
    toc2 = time.perf_counter()
    print(f"Data loading {toc1 - tic:0.4f} seconds")
    print(f"Whole time {toc2 - tic:0.4f} seconds")
    
    return clf

# clf = get_model()
# dump(clf, 'filename.joblib') #zapisany model


    