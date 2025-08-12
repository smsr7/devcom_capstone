library(ggplot2)
library(psych)
library(dplyr)
library(lattice)
library(GGally)
library(ggpubr)
library(ggfortify)
library(MASS)
library(lindia)
library(olsrr)
library(here)
library(readr)
library(tidyverse)
library(plyr)
library(scales)
library(grid)
library(car)
library(fitdistrplus)
library(EnvStats)
library(jtools)
library(officer)
library(huxtable)
library(flextable)
traindir <- "C:/Users/justi/OneDrive/Documents/College/Fall 2024/Capstone/devcom_capstone/distribution_fitting"
setwd(traindir)
df <- read.table("accuracy.csv", sep=",", header=T)

#Model
acc.lm<-lm(accuracy ~ human + night + temp + wind + vis + precip + rocky + sandy + swampy + wooded,data=df)
#Model Stats
summary(acc.lm)
AIC(acc.lm)
BIC(acc.lm)
#export model
acc.lm1<-lm(accuracy ~ human + night + temp + wind + rocky + sandy + swampy + wooded,data=df)
summ(acc.lm1)
export_summs(acc.lm1, scale = TRUE, to.file = "docx", file.name = "test.docx")

#Distribution of Accuracy based on Surface
ai <- df %>% filter(human == 0)
hum <- df %>% filter(human == 1)
airocky <- ai %>% filter(ai$surface == 'Rocky')
aigrassy <- ai %>% filter(ai$surface == "Grassy")
aiwooded <- ai %>% filter(ai$surface == "Wooded")
aiswampy <- ai %>% filter(ai$surface == "Swampy")
aisandy <- ai %>% filter(ai$surface == "Sandy")
humrocky <- hum %>% filter(hum$surface == "Rocky")
humgrassy <- hum %>% filter(hum$surface == "Grassy")
humwooded <- hum %>% filter(hum$surface == "Wooded")
humswampy <- hum %>% filter(hum$surface == "Swampy")
humsandy <- hum %>% filter(hum$surface == "Sandy")

#AI Rocky Surface
descdist(data = airocky$accuracy, discrete = FALSE, boot=1000)
#Appears uniform
eunif(airocky$accuracy, method = "mle")


#AI Grassy Surface
descdist(data = aigrassy$accuracy, discrete = FALSE, boot=1000)
#Appears uniform
eunif(aigrassy$accuracy, method = "mle")


#AI Wooded Surface
descdist(data = aiwooded$accuracy, discrete = FALSE, boot=1000)
#Appears uniform
eunif(aiwooded$accuracy, method = "mle")


#AI Swampy Surface
descdist(data = aiswampy$accuracy, discrete = FALSE, boot=1000)
#Appears uniform or Beta
eunif(aiswampy$accuracy, method = "mle")
ebeta(aiswampy$accuracy, method = "mle")


#AI Sandy Surface
descdist(data = aisandy$accuracy, discrete = FALSE, boot=1000)
#Appears uniform or Beta
eunif(aisandy$accuracy, method = "mle")
ebeta(aisandy$accuracy, method = "mle")

#Human Rocky Surface
descdist(data = humrocky$accuracy, discrete = FALSE, boot=1000)
#Appears Beta
ebeta(humrocky$accuracy, method = "mle")



#Human Grassy Surface
descdist(data = humgrassy$accuracy, discrete = FALSE, boot=1000)
#Appears uniform
eunif(humgrassy$accuracy, method = "mle")


#Human Wooded Surface
descdist(data = humwooded$accuracy, discrete = FALSE, boot=1000)
#Very bizarre, maybe Logistic?
elogis(humwooded$accuracy, method = "mle")


#Human Swampy Surface
descdist(data = humswampy$accuracy, discrete = FALSE, boot=1000)
#Appears Beta
ebeta(humswampy$accuracy, method = "mle")



#Human Sandy Surface
descdist(data = humsandy$accuracy, discrete = FALSE, boot=1000)
#Appears normal or Beta
ebeta(humsandy$accuracy, method = "mle")
enorm(humsandy$accuracy, method = "mvue")


#####Now, set up so that once UAV observes edge surface conditions, accuracy is determined by some pseudo-random number generation from distributions with found parameters.

