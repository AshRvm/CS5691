Regression : 

Classification : 
    1) Input: Training data set text file, Development data set text file 
    2) In case of real data set (or data sets with large range of values), comment out lines 109,110,248,249 and 
       uncomment the lines 113,114,252,253, since the steps used to create the dummy points for plotting pdf, decision
       boundary and constant density curves are taken as 0.1 for the first two data sets and 5 for the real data set based 
       on the range of values each dimension exhibits.
    3) All plots will be saved locally with appropriate titles, but not all plots can be shown in one execution of the program 
       since matplotlib has a limit of 20 plots per program. So, comment out one of the following plot functions, and run the
       program again : 
            1] PDFs : Line 131
            2] Constant Density Curves : Line 132
            3] Decision Boundaries : Lines 285,286
            4] Confusion Matrices : Lines 288,289
            5] ROC and DET Curves : Line 291