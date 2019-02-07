library(dplyr)
library(reshape)
library(ggplot2)
library(lubridate)

Power_Consumption <- read.csv("Dataset/Electricity Consumption Data.csv", header=TRUE)

# Re-format the datetime field into timestamp in R
Power_Consumption$Datetime <- as.Date(strptime(as.character(Power_Consumption$Datetime),format="%Y-%m-%d %H:%M:%OS"))

Power_Consumption$Datetime <- floor_date(Power_Consumption$Datetime,"month")

#Aggregate the Data by Month Now
Power_Consumption <- Power_Consumption %>% group_by(Datetime) %>%  summarise(Average_Consumption=mean(DOM_MW))

#The descriptive elements of time series are good to know before starting the visualization.

#Number of records
dim(Power_Consumption)

#Start Date & Time of Series
min(Power_Consumption$Datetime)

#End Data & Time of Series
max(Power_Consumption$Datetime)

#Interval of Recording Consumption - assuming allare equally spaced
Power_Consumption$Datetime[3]-Power_Consumption$Datetime[2]

#Highest Consumtion recorded
max(Power_Consumption$Average_Consumption)

#Lowest Consumtion recorded
min(Power_Consumption$Average_Consumption)

#Average Consumption in The Series
mean(Power_Consumption$Average_Consumption)

start_Date <- as.Date("2006-01-01")
end_Date <- as.Date("2018-01-01") 

#Simple Line Chart

ggplot(data = Power_Consumption, aes(x = Datetime, y = Average_Consumption)) +
  geom_line(color = 'red', size = 1) +
  scale_x_date(limits = c(start_Date, end_Date))+
  theme(legend.title=element_text(family="Times",size=20),
        legend.text=element_text(family="Times",face = "italic",size=15),
        plot.title=element_text(family="Times", face="bold", size=20),
        axis.title.x=element_text(family="Times", face="bold", size=12),
        axis.title.y=element_text(family="Times", face="bold", size=12)) +
    xlab("Time") +
    ylab("Power Consumption (in Mega Watt)") +
    ggtitle("Power Consmption Over Time by Month - 2005 to 2018")


## ----  ----

#Let's only focus on time series analysis on 2015 onwards only by setting axis limits

start_Date <- as.Date("2015-01-01")
end_Date <- as.Date("2017-12-31")


Power_Consumption = Power_Consumption[as.Date(Power_Consumption$Datetime) >= start_Date & as.Date(Power_Consumption$Datetime) <= end_Date,]


ggplot(data = Power_Consumption, aes(x = Datetime, y = Average_Consumption)) + 
  geom_line(color = "red", size = 1) +
  stat_smooth(  color = "blue", fill = "yellow", method = "loess" )


## ----  warning=FALSE,message=FALSE, fig.width=15, fig.height=9,fig.cap = "Figure: 9.3: Series Additive Decomposition for 2015 to 2017 Time Series Data"----

#Create a ts object for the Average Consumption with frequency as 12, i.e., 1 year.
Avg_Consumption <- ts(Power_Consumption$Average_Consumption, frequency = 12, start = c(2015, 1))

#We assume seasonal variations is constant in the period of observation, hence applying additive decomposition

Power_Consumption_Decomposed <- decompose(Avg_Consumption,type="additive")

plot(Power_Consumption_Decomposed)


## ----  ----

ar1_1 <- arima.sim(list(order = c(1,0,0), ar = 0.8), n = 25)
ar1_2 <- arima.sim(list(order = c(1,0,0), ar = -0.8), n = 25)

ar1_3 <- arima.sim(list(order = c(1,0,0), ar = 0.2), n = 25)
ar1_4 <- arima.sim(list(order = c(1,0,0), ar = -0.2), n = 25)

par(mfrow=c(2,2))
plot.ts(ar1_1, main="AR(1) with coefficient 0.8")
plot.ts(ar1_2, main="AR(1) with coefficient -0.8")
plot.ts(ar1_3, main="AR(1) with coefficient 0.2")
plot.ts(ar1_4, main="AR(1) with coefficient -0.2")
par(mfrow=c(1,1))

## ---- ----

ma1_1 <- arima.sim(list(order = c(0,0,1), ma = 0.8), n = 25)
ma1_2 <- arima.sim(list(order = c(0,0,1), ma = -0.8), n = 25)

ma1_3 <- arima.sim(list(order = c(0,0,1), ma = 0.2), n = 25)
ma1_4 <- arima.sim(list(order = c(0,0,1), ma = -0.2), n = 25)

par(mfrow=c(2,2))
plot.ts(ma1_1, main="MA(1) with coefficient 0.8")
plot.ts(ma1_2, main="MA(1) with coefficient -0.8")
plot.ts(ma1_3, main="MA(1) with coefficient 0.2")
plot.ts(ma1_4, main="MA(1) with coefficient -0.2")
par(mfrow=c(1,1))

## ----  ----

library(aTSA)

# Syntex ->  stationary.test(x, method = c("adf", "pp", "kpss"), nlag = NULL, type = c("Z_rho", "Z_tau"), lag.short = TRUE, output = TRUE)

#Augmented Dickey-Fuller test in aTSA
stationary.test(Power_Consumption$Average_Consumption, method = c("adf"), nlag = 12, type = c("Z_rho", "Z_tau"), lag.short = TRUE, output = TRUE)


library(tseries)
# Syntex -> adf.test(x, alternative = c("stationary", "explosive"),          k = trunc((length(x)-1)^(1/3)))

#Augmented Dickey-Fuller test in tseries
ts.power_consumption <- ts(Power_Consumption$Average_Consumption,frequency = 12, start=c(2015,1))

#plot(ts.power_consumption)

adf.test(ts.power_consumption, alternative="stationary", k=0)


## ----  ----

library(tseries)

#Read the Macro-economic Data
MacroData <- read.csv("Dataset/Macroeconomic Indicators Data.csv",header=TRUE)
str(MacroData)

#We will work with univariate series, so let's pick out ppi for analysis
ppi <- ts(MacroData$ppi,frequency=4,start=c(1960,1))
plot(ppi, main="Time Series Plot of Purchasing Parity Indicator(PPI)")

#Run the stationary test 
adf.test(ppi, alternative="stationary", k=0)


## ----  ----

#We will apply order 1 differencing
d.ppi <- diff(ppi, lag=1)
plot(d.ppi,main="Differenced Time Series Plot of Purchasing Parity Indicator(PPI)")

#Run the stationary test 
adf.test(d.ppi, alternative="stationary", k=0)


## ----  ----

# ACF

acf_values <- acf(Power_Consumption$Average_Consumption,type = c("correlation"))

print(acf_values)


## ----  ----

#Pacf(x, lag.max = NULL, plot = TRUE, na.action = na.contiguous,demean = TRUE, ...)

pacf_values <- pacf(Power_Consumption$Average_Consumption)

print(pacf_values)


## ----  ----

library(stats)

#ar(x, aic = TRUE, order.max = NULL,method=c("yule-walker", "burg", "ols", "mle", "yw"),na.action, series, ...)


ar_model <- ar(Power_Consumption$Average_Consumption, aic = TRUE, order.max = NULL,method=c("yule-walker"))

print(ar_model)

library(forecast)

fit <- Arima(Power_Consumption$Average_Consumption,order=c(ar_model$order,0,0))

#Plot acf plot for the residual to see if the AR model is able to handle time variations
resid <- na.omit(ar_model$resid)

acf(resid,type = c("correlation"))

#Plot the fitter Ar(12) process

plot(fit$x,col="red", type="l")
lines(fitted(fit),col="blue")



## ----  ----

library(forecast)

fit_ma <- Arima(Power_Consumption$Average_Consumption,order=c(12,0,1),method="ML")

#Plot acf plot for the residual to see if the AR model is able to handle time variations
resid <- na.omit(fit_ma$residuals)

acf(resid,type = c("correlation"))
pacf(resid,type = c("correlation"))

#Plot the fitter Ar(12) process

plot(fit_ma$x,col="red", type="l")
lines(fitted(fit_ma),col="blue")


## ----  ----

library(tseries)
library(rugarch)
library(forecast)
#analysis
adf.test(Power_Consumption$Average_Consumption) #perform stationariety test 
#model identification
acf(Power_Consumption$Average_Consumption) # lag checking and model identification
pacf(Power_Consumption$Average_Consumption) # lag checking and model identification

Box.test(Power_Consumption$Average_Consumption,lag=12,type="Ljung-Box") #check autocorrelation
Box.test(Power_Consumption$Average_Consumption^2,lag=12,type="Ljung-Box") # check arch effect

fit_arima <- arima(Power_Consumption$Average_Consumption,order=c(12,0,1),method="ML") #perform lot of precedeing task automatically
fit_arima

###Residuals Analysis 
rest<-residuals(fit_arima,standardize=TRUE)    
acf(rest)
acf(rest^2)
pacf(rest)
pacf(rest^2)
Box.test(rest,lag=12,type="Ljung-Box")
Box.test(rest^2,lag=12,type="Ljung-Box")


## ----  ----


par(mfrow = c(1,2))
fit1 <-  Arima(Power_Consumption$Average_Consumption, order = c(12,0,1), 
             include.drift = T,method="ML")
future <-  forecast(fit1, h = 12)
plot(future)
fit2 <- Arima(Power_Consumption$Average_Consumption, order = c(12,0,1), 
             include.drift = F,method="ML")
future2 <- forecast(fit2, h = 12)
plot(future2)


## ----  ----

library(astsa)
library(lmtest)

#Load the sales data
SalesData <- read.csv("Dataset/Super Market Sales Data.csv",header=TRUE)
str(SalesData)

#Now we fit a linear regression model
fit_linear <- lm(StoreSales~ RegionSales,data=SalesData)
summary(fit_linear)

#Durbin Watson Test to detect auto-correlation
dwtest(fit_linear)

#Autocorrelation Function
acf(fit_linear$residuals,type = c("correlation"))


## ----  ----

library(orcutt)


#Now we apply Cochrane Orcutt Correction to the linear regression model
CO_fit_linear <- cochrane.orcutt(fit_linear)
CO_fit_linear

#Durbin Watson Test to detect auto-correlation
dwtest(CO_fit_linear$Cochrane.Orcutt)

#Autocorrelation Function
acf(CO_fit_linear$Cochrane.Orcutt$residuals,type = c("correlation"))


