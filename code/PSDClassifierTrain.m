function[whi, woh] = PSDClassifierTrain(trainData, trainLabels, validData, validLabels, testData, testLabels, numIter, numhid)

fprintf('setting up network... \n');

inlen = size(trainData,1);
numTrainImages = size(trainData,2);
numTestImages = size(testData, 2);
numValidImages = size(validData, 2);
targlen = 2;

whi = (normrnd(0,1/sqrt(inlen+1),[numhid,inlen+1]));    %randomized weight matrix for input layer to hidden layer                 
woh = (normrnd(0,1/sqrt(numhid+1),[targlen,numhid+1]));  %randomized weight matrix for hidden layer to output layer
learnho = 0.001;
learnih = 0.005;
a = .9;         %momentum constant
dwoldh = 0;     %old weight change matrix for input to hidden units  
dwoldo = 0;     %old weight change matrix for hidden to output units

run = true;
epoch = 0;
%% Stochastic Gradient Descent Code

Error = [];

ValidPercentWrong = [];

TestPercentWrong = [];

%mainprogram

%% training
fprintf('training... \n');

while run
    
    epoch = epoch + 1;
    
    trainError = 0;
    validWrong = 0;
    testWrong = 0;

    fprintf('epoch %i... \n', epoch);
    
    %% do SGD for training images
    for i = randperm(numTrainImages)
        input = [1; trainData(:,i)];
        
        if trainLabels(i) == 0
            targ = [1;0];
        elseif trainLabels(i) == 1
            targ = [0;1];
        else
            fprintf('error, target allocated incorrectly\n');
            return;
        end
        
        %forward propagate function
        neti = [whi*input];
        hout = [1./(1+exp(-neti))];
        
        h_layer = [1; hout];
        neto = woh*[h_layer];
        
        out = 1./(1+exp(-neto));
        oprime = out.*(1-out);
        
        hprime = hout.*(1-hout);
        
        deltao = (targ - out).*oprime;
        deltah = hprime.*(woh(:,2:numhid+1)'*deltao);
        
        dwo = learnho.*(deltao*[h_layer]') + dwoldo*a;
        woh = woh + dwo;
        
        dwh = learnih.*((deltah)*input') + dwoldh*a;
        whi = whi + dwh;
        
        trainError = trainError + 0.5*sum((targ-out).^2)/numTrainImages; 
        
    end
    %% print training error
    Error = [Error, trainError];
    fprintf('    training error: %f \n', Error(epoch));
    
    
    %% forward propagate and find out validation error

    for i = randperm(numValidImages)
        wrong = feedforwards(whi, woh, validData(:,i), validLabels(i));
        validWrong = validWrong + wrong;
    end  
    
    %% print validation percent wrong
    vpWrong = 100*(validWrong/numValidImages);
    ValidPercentWrong = [ValidPercentWrong, vpWrong];
    fprintf('    validation percent wrong: %%%f \n', ValidPercentWrong(epoch));
    
    %% forwad propagate and find out test error

    for i = randperm(numTestImages)
        wrong = feedforwards(whi, woh, testData(:,i), testLabels(i));
        testWrong = testWrong + wrong;
    end
    
    %% print test percent wrong
    tpWrong = 100*(testWrong/numTestImages);
    TestPercentWrong = [TestPercentWrong, tpWrong];
    fprintf('    test percent wrong: %%%f \n', TestPercentWrong(epoch));
    
    %% break when...
    if(epoch == numIter)
        run = false;
        fprintf('****************TRAINING ENDS*********************')
        plot(Error, 'b'); hold;
        plot(ValidPercentWrong, 'r');
        plot(TestPercentWrong, 'g');
        legend('Training: SSE', 'Validation: Percent Wrong', 'Test: Percent Wrong');
    end

end






    