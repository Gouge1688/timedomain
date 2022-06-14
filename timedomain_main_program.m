clc,clear,close all;  
%X = load('Sound.txt');%上载一个声音信号。
X = load('Sound.mat');X=X.X;
X=X-mean(X);               %去直流分量 (有必要)  ……也称之为去趋势项。
Fs = 40000;                % Sampling frequency                    
T = 1/Fs;                  % Sampling period       

A=X(600001:750000);         % 选取未熔透状态中20W个点    
B=X(1900001:2050000);
C=X(2500001:2650000);

%低通滤波组合
filter_lowpass1 = fir1(128,0.99);%根据经验将截止频率设为0.65-0.99更加精确，取到了0到0.65*20000hz的频率范围
filter_lowpass2 = fir1(128,0.025);
X1 = filter(filter_lowpass1,1,X);
X2 = filter(filter_lowpass2,1,X);
X=X1-X2;                              %目的：通过低通滤波去除500HZ以下的频率
%↓↓↓↓↓小波阈值去噪 
A1 = wdenoise(A,4, 'Wavelet', 'db3', 'DenoisingMethod', 'UniversalThreshold', 'ThresholdRule', 'Soft', 'NoiseEstimate', 'LevelIndependent');
B1 = wdenoise(B,4, 'Wavelet', 'db3', 'DenoisingMethod', 'UniversalThreshold', 'ThresholdRule', 'Soft', 'NoiseEstimate', 'LevelIndependent');
C1 = wdenoise(C,4, 'Wavelet', 'db3', 'DenoisingMethod', 'UniversalThreshold', 'ThresholdRule', 'Soft', 'NoiseEstimate', 'LevelIndependent');
A=A1; B=B1; C=C1;       %小波包阈值降噪后的幅值 db3，4层分解，来源于论文对语音信号的降噪的结论
A_fenzhen=fenzhen(A,1024,512);A=A_fenzhen';A(:,end)=[];   
B_fenzhen=fenzhen(B,1024,512);B=B_fenzhen';B(:,end)=[];   
C_fenzhen=fenzhen(C,1024,512);C=C_fenzhen';C(:,end)=[];
L = length(A);                %%每一帧信号的长度，即窗长
t = (0:L-1)*T;                % Time vector
%计算每一帧的特征 
n=291; %样本个数
N=size(A,1);%样本点数,就是每一帧里的样点数
 for j=1:n
       %X(:,j)=abs(hilbert(x(j,:)));
       %%这个值后边需要探究一下，有什么用；经过探究，经过hilbert后相当于是对信号的包络进行特征的提取
       timeA(j,:)=time_statistical_compute(A(:,j));%%%%%%%多维矩阵计算时域特征矩阵
       timeB(j,:)=time_statistical_compute(B(:,j));
       timeC(j,:)=time_statistical_compute(C(:,j));
 end
 for i = 1:16
timeA(:,i) = smooth(timeA(:,i),40,'sgolay',10);     %经测试，平滑后效果能够提升将近15%。
timeB(:,i) = smooth(timeB(:,i),40,'sgolay',10); 
 timeC(:,i) = smooth(timeC(:,i),40,'sgolay',10); 
 end
 
 %↓↓↓↓以下进行 每一帧参量的对比
figure;
subplot(421);plot(timeA(:,1)); hold on ;plot(timeB(:,1)); hold on ;plot(timeC(:,1))   ;title( '%均值')
subplot(422);plot(timeA(:,2)); hold on ;plot(timeB(:,2)); hold on ;plot(timeC(:,2))   ;title( ' %均方根值')
subplot(423);plot(timeA(:,3)); hold on ;plot(timeB(:,3)); hold on ;plot(timeC(:,3))   ;title( ' %方根幅值')
subplot(424);plot(timeA(:,4)); hold on ;plot(timeB(:,4)); hold on ;plot(timeC(:,4))    ;title( '%绝对平均值')
subplot(425);plot(timeA(:,5)); hold on ;plot(timeB(:,5)); hold on ;plot(timeC(:,5))    ;title( '%偏斜度')
subplot(426);plot(timeA(:,6)); hold on ;plot(timeB(:,6)); hold on ;plot(timeC(:,6))    ;title( '%峭度')
subplot(427);plot(timeA(:,7)); hold on ;plot(timeB(:,7)); hold on ;plot(timeC(:,7))    ;title( '%方差')
subplot(428);plot(timeA(:,8)); hold on ;plot(timeB(:,8)); hold on ;plot(timeC(:,8))    ;title( '%最大值')
figure;
subplot(421);plot(timeA(:,9)); hold on ;plot(timeB(:,9)); hold on ;plot(timeC(:,9));   title( '%最小值')
subplot(422);plot(timeA(:,10)); hold on ;plot(timeB(:,10)); hold on ;plot(timeC(:,10));title( '%峰峰值')
subplot(423);plot(timeA(:,11)); hold on ;plot(timeB(:,11)); hold on ;plot(timeC(:,11));title( '%波形指标')
subplot(424);plot(timeA(:,12)); hold on ;plot(timeB(:,12)); hold on ;plot(timeC(:,12));title( '%峰值指标')
subplot(425);plot(timeA(:,13)); hold on ;plot(timeB(:,13)); hold on ;plot(timeC(:,13));title( '%脉冲指标')
subplot(426);plot(timeA(:,14)); hold on ;plot(timeB(:,14)); hold on ;plot(timeC(:,14));title( '%裕度指标')
subplot(427);plot(timeA(:,15)); hold on ;plot(timeB(:,15)); hold on ;plot(timeC(:,15));title( '%偏斜度指标')
subplot(428);plot(timeA(:,16)); hold on ;plot(timeB(:,16)); hold on ;plot(timeC(:,16));title( '%峭度指标')%经过好几次测试，发现该特征对声音信号的时域识别效果很拉胯，可考虑去掉

time=timeA;   %以后这种赋值就别放在for循环里边了，会打破人家里边的变量循环
for i = 1:n
time(i+n,:)=timeB(i,:);
time(i+n*2,:)=timeC(i,:);
end
time(:,[5,16])=[];%删除掉第5，第16列的特征，因为其效果很差劲

D=ones(n,1);
for i = 1:n
D(i+n,:)=D(i,:)*2;
D(i+n*2,:)=D(i,:)*3;
end
for i = 1:length(time(1,:))  %length(time(1,:))代表时域矩阵的列的数量
time(:,i) = smooth(time(:,i),20,'sgolay',10);     %经测试，平滑后效果能够提升将近15%。
end
feature=time;
feature(:,end+1)=D;



%% 读取数据
data = feature;
matrix = data(:,1:end-1);           
label = data(:,end); 
classnumber=3;
%%
% 1. 随机产生训练集和测试集
n = randperm(size(matrix,1));  %size(matrix,1)里的1是维度dim的意思 size的长度就是matrix的长度
% 2. 训练集――训练多少个样本
train_matrix = matrix(n(1: round(length(matrix)*0.8)),:);  %round 是取整函数
train_label = label(n(1: round(length(matrix)*0.8)),:);
            train_data=train_matrix;
            train_data_labels=train_label;
 % 3. 测试集――26个样本
test_matrix = matrix(n(round(length(matrix)*0.8)+1:end),:);
test_label = label(n(round(length(matrix)*0.8)+1:end),:);
             test_data=test_matrix;
             test_data_labels=test_label;
             [train_scale,test_scale,ps] = scaleForSVM(train_data,test_data,0,1); %这就是一个归一化
            %% 粒子群参数
             pso_option.c1 = 1.5; 
             pso_option.c2 = 1.7; 
             pso_option.maxgen = 30; 
             pso_option.sizepop = 10;    %这个值不能太大，会很花时间，一般为20到40，简单问题10就足够  
             pso_option.k = 0.6;        %这个值用管
             pso_option.wV = 1; 
             pso_option.wP = 1; 
             pso_option.v = 3; 
             pso_option.popcmax = 2^10; 
             pso_option.popcmin = 2^-10;
             pso_option.popgmax = 1000; 
             pso_option.popgmin = 2^-10;
             %% 寻优C G参数 
[bestacc,bestc,bestg]=Pso_for_SVM(train_data_labels,train_scale,pso_option);
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];

%%
% 2. 创建/训练SVM模型
             train_label=train_data_labels;        %4.1加上的
             Train_matrix=train_scale;             %4.1加上的,这里加归一化之后的就没问题
model = libsvmtrain(train_label,Train_matrix,cmd);
             Test_matrix=test_scale; 
 %% V. SVM仿真测试
[predict_label_1,accuracy_1,decision_values1] = libsvmpredict(train_label,Train_matrix,model);
[predict_label_2,accuracy_2,decision_values2] = libsvmpredict(test_label,Test_matrix,model);
result_1 = [train_label predict_label_1];
result_2 = [test_label predict_label_2];
%% VI. 绘图
figure
plot(1:length(test_label),test_label,'r*')
hold on
plot(1:length(test_label),predict_label_2,'bo')
grid on
xlabel('测试集样本编号')
ylabel('测试集样本类别')

legend('真实类别','预测类别')
string = {'测试集SVM预测结果对比(RBF核函数)';
        ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string);