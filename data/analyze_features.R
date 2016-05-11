prices = read.table(file = 'prices2013.dat',header=T)

features = data.frame(prices$HolidayFlag,prices$WeekOfYear,prices$DayOfWeek,prices$PeriodOfDay,prices$ForecastWindProduction,prices$SystemLoadEA,prices$SMPEA,prices$ORKTemperature,prices$ORKWindspeed,prices$CO2Intensity)
y = prices$SMPEP2
cor(features,y,use='complete.obs')

#y_prev = c(rep(0,24),head(prices$SMPEP2,-24))

#daysums = colSums(matrix(y_prev, nrow=24))/24
#y_sumdayprev = rep(daysums,each=24)


sum(prices$SMPEP2 > 300)

#,y_sumdayprev)#,prices$ORKTemperature,prices$ORKWindspeed,prices$CO2Intensity)
hist(prices$SMPEP2,breaks=100)
shapiro.test(prices$SMPEP2[1:5000])
