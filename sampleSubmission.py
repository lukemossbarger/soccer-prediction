import pandas as pd
import numpy as np

def calc_my_probs(df):
    """Calculate match outcome probabilities based on team quality metrics"""
    # Calculate home team quality metric
    home_quality_metric = (df['HG_TY'] + df['HG_LY'] - df['HGA_TY'] - df['HGA_LY']) / (df['HM_TY'] + df['HM_LY'])
    
    # Calculate away team quality metric  
    away_quality_metric = (df['AG_TY'] + df['AG_LY'] - df['AGA_TY'] - df['AGA_LY']) / (df['AM_TY'] + df['AM_LY'])
    
    # Handle NaN values by setting them to 0
    home_quality_metric = home_quality_metric.fillna(0)
    away_quality_metric = away_quality_metric.fillna(0)
    
    # Calculate team difference quality metric
    team_difference_quality_metric = home_quality_metric - away_quality_metric
    
    # Set draw probability to 0.25
    df['dProb'] = 0.25
    
    # Calculate home win probability using logistic function
    df['hProb'] = (1 - df['dProb']) * np.exp(team_difference_quality_metric) / (1 + np.exp(team_difference_quality_metric))
    
    # Calculate away win probability
    df['aProb'] = 1 - df['hProb'] - df['dProb']
    
    return df

def calc_my_probs_half_time(df):
    """Calculate halftime-adjusted probabilities"""
    # Calculate quality metrics (same as above)
    home_quality_metric = (df['HG_TY'] + df['HG_LY'] - df['HGA_TY'] - df['HGA_LY']) / (df['HM_TY'] + df['HM_LY'])
    away_quality_metric = (df['AG_TY'] + df['AG_LY'] - df['AGA_TY'] - df['AGA_LY']) / (df['AM_TY'] + df['AM_LY'])
    
    home_quality_metric = home_quality_metric.fillna(0)
    away_quality_metric = away_quality_metric.fillna(0)
    
    # Adjust quality metric with halftime score difference
    team_difference_quality_metric = home_quality_metric - away_quality_metric + df['Hgoals1H'] - df['Agoals1H']
    
    # Set draw probability based on halftime score
    df['dProbHT'] = np.where(df['Hgoals1H'] == df['Agoals1H'], 0.35, 0.15)
    
    # Note: Original R code has a bug - it uses df$dProb instead of df$dProbHT
    # I'll fix this in the Python version
    df['hProbHT'] = (1 - df['dProbHT']) * np.exp(team_difference_quality_metric) / (1 + np.exp(team_difference_quality_metric))
    
    # Calculate away probability
    df['aProbHT'] = 1 - df['hProbHT'] - df['dProbHT']
    
    return df

def calc_my_goals_second_half(df):
    """Calculate predicted goals for second half"""
    # Calculate average scoring rates
    home_average_scoring = (df['HG_TY'] + df['HG_LY']) / (df['HM_TY'] + df['HM_LY'])
    away_average_scoring = (df['AG_TY'] + df['AG_LY']) / (df['AM_TY'] + df['AM_LY'])
    
    # Handle NaN values
    home_average_scoring = home_average_scoring.fillna(1.25)
    away_average_scoring = away_average_scoring.fillna(1.25)
    
    # Calculate predicted goals for second half
    df['predGoals2H'] = (0.50 + 
                        0.10 * df['Hgoals1H'] + 
                        0.10 * df['Agoals1H'] + 
                        0.50 * home_average_scoring + 
                        0.50 * away_average_scoring)
    
    return df

def main():
    """Main function to run the soccer prediction pipeline"""
    # Read the CSV file
    df = pd.read_csv("soccermatches2.csv")
    
    # Apply all calculation functions
    df = calc_my_probs(df)
    df = calc_my_probs_half_time(df)
    df = calc_my_goals_second_half(df)
    
    # Write output to CSV
    df.to_csv("samplename_output.csv", index=False)
    
    print("Predictions calculated and saved to samplename_output.csv")
    return df

if __name__ == "__main__":
    main()
