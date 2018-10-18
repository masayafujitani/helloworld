##support vector machine

##出力されるファイル
##1_parameter_tune_result.pdf    :パラメータ最適化結果
##2_prediction_result.dat        :予測結果（LOOCVの結果）
##3_prediction_result_table.dat  :2のtable Ver.
##4_best_model_supprt_vector.dat :bestmodelのサポートベクター

library(foreach)
library(doParallel)

cl <- makeCluster(14)
registerDoParallel(cl)

a <- foreach(i=1:14) %dopar% {

categorys <- i

####################################################手入力
##データ 1列目:サンプル名 2列目以降:データ
reads1 <- paste("result/logistic_707scale_cat14/category",categorys,"/category",categorys,"_x.dat",sep="")

##↑変数ごとに標準化してあるか 1:Yes 2:No
sc <- 1

##教師値 1列目:サンプル名 2列目:教師値
reads2 <- paste("result/logistic_707scale_cat14/category",categorys,"/category",categorys,"_y.dat",sep="")

##結果を出力するフォルダ名
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

##プログラム確認用のテストデータ
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

##パラメータの最適化
tune <- tune.svm(x,                                                      ##データ
                 y,                                                      ##教師値
                 gamma=10^(seq(-5,5,0.1)),                               ##gamma 10^-5から10^5まで0.1ずつ変化させる（グリッドサーチ）
                 cost=10^(seq(-2,2,0.1)),                                ##C 10^-2から10^2まで0.1ずつ変化させる（グリッドサーチ）
                 tunecontrol=tune.control(sampling="cross",cross=10)     ##@foldクロスバリデーションにてgammaとCを決定
                 )


#パラメータの可視化
pdf(paste(dirname1,"/1_parameter_tune_result.pdf",sep=""))
plot(tune,transform.x=log10,transform.y=log10,
     main=paste("gamma:",signif(tune$best.parameters$gamma,4),
                " cost:",signif(tune$best.parameters$cost,4),
                " accuracy:",signif(100-tune$best.performance*100,4),sep=""))
dev.off()
##最適化を2回する方法もあるとのこと
##1回目で大まかに範囲を決定して2回目で決めるらしい


##最適なパラメータを使って予測(LOOCVにてモデルの性能を確認)
for(i in 1:nrow(x)){
	
	trainX <- x[-i,]
	trainY <- y[-i]
	validX <- rbind(x[i,],x[i,]) ##なぜか1行だとpredictでエラーになるので2行にする
	validY <- y[i]
	
	##モデル作成
	model <- svm(trainX,                            #データ
                 trainY,                            #教師値
                 method="C-classification",         #分類方法 C-classification, nu-classification, one-classification, eps-regression, nu-regression
                 kernel="radial",                   #カーネル関数 radial, linear, polynomial, sigmoid
                 gamma=tune$best.parameters$gamma,  #最適なgamma
                 cost=tune$best.parameters$cost     #最適なC
                 #,class.weights=@@@                 #weight 群の数が全然違う時は@@@を書き換える
                 )
	
	resultbox[i,3] <- as.character(predict(model,validX))[1]
	resultbox[i,2] <- as.character(validY)
	
	}

##もしくはこちら
#model <- svm(x,                                 #データ
#             y,                                 #教師値
#             method="C-classification",         #分類方法 C-classification, nu-classification, one-classification, eps-regression, nu-regression
#             kernel="radial",                   #カーネル関数 radial, linear, polynomial, sigmoid
#             gamma=tune$best.parameters$gamma,  #最適なgamma
#             cost=tune$best.parameters$cost,    #最適なC
#             cross=10
#             )
#aa <- result$Total.accuracy


##精度算出
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
best.model <- svm(x,                                 #データ
                  y,                                 #教師値
                  method="C-classification",         #分類方法 C-classification, nu-classification, one-classification, eps-regression, nu-regression
                  kernel="radial",                   #カーネル関数 radial, linear, polynomial, sigmoid
                  gamma=tune$best.parameters$gamma,  #最適なgamma
                  cost=tune$best.parameters$cost,    #最適なC
                  )

write.table(best.model$SV,paste(dirname1,"/4_best_model_support_vector.dat",sep=""),sep="\t",row.names=F,col.names=T)

##係数
w = t(best.model$coefs) %*% best.model$SV
write.table(w,paste(dirname1,"/5_best_model_coefficience.dat",sep=""),sep="\t",row.names=F,col.names=T)


}
stopCluster(cl)

