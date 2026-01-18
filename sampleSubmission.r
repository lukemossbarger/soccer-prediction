df <- read.csv("soccermatches2.csv")

calcMyProbs <- function(df){
  homeQualityMetric <- with (df, (HG_TY+HG_LY-HGA_TY-HGA_LY)/(HM_TY+HM_LY))
  awayQualityMetric <- with (df, (AG_TY+AG_LY-AGA_TY-AGA_LY)/(AM_TY+AM_LY))
  homeQualityMetric[is.na(homeQualityMetric)] <- 0
  awayQualityMetric[is.na(awayQualityMetric)] <- 0
  teamDifferenceQualityMetric <- homeQualityMetric - awayQualityMetric
  df$dProb <- 0.25
  df$hProb <- (1 - df$dProb) * exp(teamDifferenceQualityMetric) / (1 + exp(teamDifferenceQualityMetric))
  df$aProb <- 1 - df$hProb - df$dProb
  df
}

calcMyProbsHalfTime <- function(df){
  homeQualityMetric <- with (df, (HG_TY+HG_LY-HGA_TY-HGA_LY)/(HM_TY+HM_LY))
  awayQualityMetric <- with (df, (AG_TY+AG_LY-AGA_TY-AGA_LY)/(AM_TY+AM_LY))
  homeQualityMetric[is.na(homeQualityMetric)] <- 0
  awayQualityMetric[is.na(awayQualityMetric)] <- 0
  teamDifferenceQualityMetric <- homeQualityMetric - awayQualityMetric + df$Hgoals1H - df$Agoals1H
  df$dProbHT <- with (df, ifelse(Hgoals1H==Agoals1H, 0.35, 0.15))
  df$hProbHT <- (1 - df$dProb) * exp(teamDifferenceQualityMetric) / (1 + exp(teamDifferenceQualityMetric))
  df$aProbHT <- 1 - df$hProb - df$dProb
  df
}

calcMyGoalsSecondHalf <- function(df){
    homeAverageScoring <- with (df, (HG_TY+HG_LY)/(HM_TY+HM_LY))
    awayAverageScoring <- with (df, (AG_TY+AG_LY)/(AM_TY+AM_LY))
    homeAverageScoring[is.na(homeAverageScoring)] <- 1.25
    awayAverageScoring[is.na(awayAverageScoring)] <- 1.25
    df$predGoals2H <- 0.50 + 0.10 * df$Hgoals1H + 0.10 * df$Agoals1H + 0.50 * homeAverageScoring + 0.50 * awayAverageScoring
    df
}

df <- calcMyProbs(df)
df <- calcMyProbsHalfTime(df)
df <- calcMyGoalsSecondHalf(df)

write.csv(df, "samplename_output.csv", row.names=FALSE)