prices = read.table(file = 'prices2013.dat',header=T)

prices$PeriodsToPeak = abs(((prices$PeriodOfDay-12)%%48)-24)

write.table(prices,file='newprices2013.dat',col.names= F, row.names = F,na = "NaN")
#extra10h = abs(((prices$PeriodOfDay-4)%%48)-24)
features = data.frame(prices$PeriodsToPeak,prices$HolidayFlag,prices$WeekOfYear,prices$DayOfWeek,prices$PeriodOfDay,prices$ForecastWindProduction,prices$SystemLoadEA,prices$SMPEA,prices$ORKTemperature,prices$ORKWindspeed,prices$CO2Intensity)
y = prices$SMPEP2
cor(features,y,use='complete.obs')
cor(prices$PeriodsToPeak,prices$SMPEP2,use='complete.obs')

new = data.frame(prices,extra18h)
write.table(new,file="newprices2013.dat")
#y_prev = c(rep(0,24),head(prices$SMPEP2,-24))

#daysums = colSums(matrix(y_prev, nrow=24))/24
#y_sumdayprev = rep(daysums,each=24)

pricesperperiod = rowSums(matrix(prices$SMPEP2,nrow=48))/(38016/48)
plot(1:48*.5,pricesperperiod,xlab="Timeslot (hour)",ylab = "Average price")
max(pricesperperiod)


new_feature = rep(pricesperperiod,38016/48)

plot(prices$PeriodOfDay,prices$SMPEP2)
plot(prices$PeriodOfDay,prices$SMPEA)
plot(prices$SMPEA,prices$SMPEP2)
plot(prices$SMPEP2,new_feature)
cor(prices$SMPEP2,new_feature,use='complete.obs')

sum(prices$SMPEP2 > 300)

#,y_sumdayprev)#,prices$ORKTemperature,prices$ORKWindspeed,prices$CO2Intensity)
hist(prices$SMPEP2,breaks=100)
shapiro.test(prices$SMPEP2[1:5000])
