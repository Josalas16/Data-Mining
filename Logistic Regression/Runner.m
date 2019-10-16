
Test_Data =  dlmread('test_data.txt');
Test_Labels = dlmread('test_labels.txt');
Train_Data = dlmread('training_data.txt');
Train_Labels = dlmread('training_labels.txt');


coef = glmfit(Train_Data, Train_Labels, 'binomial', 'link', 'logit');

pre = LogisticRegression(Test_Data, coef);

[TP, FP, TN, FN] = confusionmat(pre, Test_Labels);

% Determining if its risky or not
Risky = TP;
False_NotRisky = FN;
True_NotRisky = TN;
False_Risky = FP;

% Function to determing whether its risky or not, test and labels
function [TP, FP, TN, FN] = confusionmat(Test, Labels)
%Starting values are zeros
    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0;
    %Testing for the values wiht the labels to see if its risky or not
    for i = 1 : length(Test)
        if Test(i) == Labels(i) && Labels(i) == 1
            TP = TP + 1;
        elseif Test(i) == 1 && Labels(i) ~= Test(i)
            FP = FP + 1;
        elseif Test(i) == Labels(i) && Labels(i) == 0
            TN = TN + 1;
        elseif Test(i) == 0 && Labels(i) ~= Test(i)
            FN= FN + 1;
        end
    end
end

%Function for the logisticRegression
%Using of Test Data and coefficients
function [pre] = LogisticRegression(Test_Data, coef)
    coef = transpose(coef);
    for i = 1 : length(Test_Data)
        z = coef(1) + sum(times(coef(2:end), Test_Data(i, 1:end)));
        z = 1./ (1 + exp(-z));
        if z > 0.5
            pre(i) = 1;
        else
            pre(i) = 0;
        end
    end
    pre = transpose(pre);
end


