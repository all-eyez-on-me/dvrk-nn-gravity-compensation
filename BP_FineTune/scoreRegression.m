function [MSE,RMSE,R2,MAE,MAPE,RPD] = scoreRegression(testY,predictY)
diff = testY-predictY;
absdiff = abs(diff);
diff2 = diff.^2;
[m,n] = size(testY);
MSE = sum(diff2(:))/m;
RMSE = sqrt(MSE);
MAE = sum(absdiff(:))/m;

meanTestY = mean(testY);
diffTestY = testY-meanTestY;
diffTestY2 = diffTestY.^2;
R2 = 1-sum(diff2(:))/sum(diffTestY2(:));

err = abs(diff./predictY);
MAPE = sum(err(:))*100/m;

RPD = std(predictY)/RMSE;
if RPD>1e5
    RPD = 0;
end


