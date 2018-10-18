##support vector machine

##�o�͂����t�@�C��
##1_parameter_tune_result.pdf    :�p�����[�^�œK������
##2_prediction_result.dat        :�\�����ʁiLOOCV�̌��ʁj
##3_prediction_result_table.dat  :2��table Ver.
##4_best_model_supprt_vector.dat :bestmodel�̃T�|�[�g�x�N�^�[

library(foreach)
library(doParallel)

cl <- makeCluster(14)
registerDoParallel(cl)

a <- foreach(i=1:14) %dopar% {

categorys <- i

####################################################�����
##�f�[�^ 1���:�T���v���� 2��ڈȍ~:�f�[�^
reads1 <- paste("result/logistic_707scale_cat14/category",categorys,"/category",categorys,"_x.dat",sep="")

##���ϐ����ƂɕW�������Ă��邩 1:Yes 2:No
sc <- 1

##���t�l 1���:�T���v���� 2���:���t�l
reads2 <- paste("result/logistic_707scale_cat14/category",categorys,"/category",categorys,"_y.dat",sep="")

##���ʂ��o�͂���t�H���_��
dirname1 <- paste("result/svm/707scale_category",categorys,sep="")

####################################################

if (!require('e1071')) install.packages('e1071')
library(e1071)

dir.create(dirname1)

data1 <- read.table(reads1,header=T)
data2 <- read.table(reads2,header=T)

if(sc==1) x <- data1[,-1]
if(sc==2) x <- scale(data1[,-1])
y <- data2[,1]

##�v���O�����m�F�p�̃e�X�g�f�[�^
#data(iris)

#2group
#x <- scale(iris[1:100,-5])
#y <- iris[1:100,5]

#3group
#x <- scale(iris[,-5])
#y <- iris[,5]

#data1 <- matrix(paste("sample",1:nrow(x),sep=""),nrow(x),1)

resultbox <- matrix(0,nrow(x),3)
resultbox[,1] <- as.character(data1[,1])

y <- as.factor(y)

##�p�����[�^�̍œK��
tune <- tune.svm(x,                                                      ##�f�[�^
                 y,                                                      ##���t�l
                 gamma=10^(seq(-5,5,0.1)),                               ##gamma 10^-5����10^5�܂�0.1���ω�������i�O���b�h�T�[�`�j
                 cost=10^(seq(-2,2,0.1)),                                ##C 10^-2����10^2�܂�0.1���ω�������i�O���b�h�T�[�`�j
                 tunecontrol=tune.control(sampling="cross",cross=10)     ##@fold�N���X�o���f�[�V�����ɂ�gamma��C������
                 )


#�p�����[�^�̉���
pdf(paste(dirname1,"/1_parameter_tune_result.pdf",sep=""))
plot(tune,transform.x=log10,transform.y=log10,
     main=paste("gamma:",signif(tune$best.parameters$gamma,4),
                " cost:",signif(tune$best.parameters$cost,4),
                " accuracy:",signif(100-tune$best.performance*100,4),sep=""))
dev.off()
##�œK����2�񂷂���@������Ƃ̂���
##1��ڂő�܂��ɔ͈͂����肵��2��ڂŌ��߂�炵��


##�œK�ȃp�����[�^���g���ė\��(LOOCV�ɂă��f���̐��\���m�F)
for(i in 1:nrow(x)){
	
	trainX <- x[-i,]
	trainY <- y[-i]
	validX <- rbind(x[i,],x[i,]) ##�Ȃ���1�s����predict�ŃG���[�ɂȂ�̂�2�s�ɂ���
	validY <- y[i]
	
	##���f���쐬
	model <- svm(trainX,                            #�f�[�^
                 trainY,                            #���t�l
                 method="C-classification",         #���ޕ��@ C-classification, nu-classification, one-classification, eps-regression, nu-regression
                 kernel="radial",                   #�J�[�l���֐� radial, linear, polynomial, sigmoid
                 gamma=tune$best.parameters$gamma,  #�œK��gamma
                 cost=tune$best.parameters$cost     #�œK��C
                 #,class.weights=@@@                 #weight �Q�̐����S�R�Ⴄ����@@@������������
                 )
	
	resultbox[i,3] <- as.character(predict(model,validX))[1]
	resultbox[i,2] <- as.character(validY)
	
	}

##�������͂�����
#model <- svm(x,                                 #�f�[�^
#             y,                                 #���t�l
#             method="C-classification",         #���ޕ��@ C-classification, nu-classification, one-classification, eps-regression, nu-regression
#             kernel="radial",                   #�J�[�l���֐� radial, linear, polynomial, sigmoid
#             gamma=tune$best.parameters$gamma,  #�œK��gamma
#             cost=tune$best.parameters$cost,    #�œK��C
#             cross=10
#             )
#aa <- result$Total.accuracy


##���x�Z�o
tab <- table(resultbox[,2],resultbox[,3])
acc <- matrix(0,1,3)
acc[1,1] <- "accuracy"
acc[1,2] <- "="
acc[1,3] <- sum(tab[row(tab)==col(tab)])/sum(tab)
resultbox2 <- rbind(resultbox,acc)
colnames(resultbox2) <- c("label","true","predict")
write.table(resultbox2,paste(dirname1,"/2_prediction_result.dat",sep=""),sep="\t",row.names=F,col.names=T)

tab2 <- cbind(rownames(tab),tab)
colnames(tab2) <- c(paste("accuracy = ",acc[1,3],sep=""),colnames(tab))
write.table(tab2,paste(dirname1,"/3_prediction_result_table.dat",sep=""),sep="\t",row.names=F,col.names=T)


##best model
best.model <- svm(x,                                 #�f�[�^
                  y,                                 #���t�l
                  method="C-classification",         #���ޕ��@ C-classification, nu-classification, one-classification, eps-regression, nu-regression
                  kernel="radial",                   #�J�[�l���֐� radial, linear, polynomial, sigmoid
                  gamma=tune$best.parameters$gamma,  #�œK��gamma
                  cost=tune$best.parameters$cost,    #�œK��C
                  )

write.table(best.model$SV,paste(dirname1,"/4_best_model_support_vector.dat",sep=""),sep="\t",row.names=F,col.names=T)

##�W��
w = t(best.model$coefs) %*% best.model$SV
write.table(w,paste(dirname1,"/5_best_model_coefficience.dat",sep=""),sep="\t",row.names=F,col.names=T)


}
stopCluster(cl)
