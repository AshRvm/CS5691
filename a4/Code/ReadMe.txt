LogReg_SVM.py :
    run the file normally

ANN.py : 
    Select which dataset to run the function on, by uncommenting the corresponding function call in the main function :
        ANN_Syn()
        ANN_Image()
        ANN_Handwritten()
        ANN_Digits()
    
    For each dataset, choose whether to apply PCA or LDA or neither by uncommenting the corresponding 
    sections in the corresponding function calls for the particular dataset
        For e.g, for the image dataset, uncomment the sections starting from lines 388 and 432, and comment
        the section starting from line 424, for running ANN after PCA on the image dataset

KNN.py :
    Similar to ANN.py