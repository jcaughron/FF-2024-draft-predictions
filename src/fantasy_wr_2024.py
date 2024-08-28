"""## Player fantasy stats"""

import pandas as pd
fantasy_stats_2023 = pd.read_csv('Fantasy_player_stats_2023.csv')
fantasy_stats_2022 = pd.read_csv('Fantasy_player_stats_2022.csv')
fantasy_stats_2021 = pd.read_csv('Fantasy_player_stats_2021.csv')
fantasy_stats_2020 = pd.read_csv('Fantasy_player_stats_2020.csv')
fantasy_stats_2019 = pd.read_csv('Fantasy_player_stats_2019.csv')


player_stats_2023 = clean_player_stats(fantasy_stats_2023)
player_stats_2022 = clean_player_stats(fantasy_stats_2022)
player_stats_2021 = clean_player_stats(fantasy_stats_2021)
player_stats_2020 = clean_player_stats(fantasy_stats_2020)
player_stats_2019 = clean_player_stats(fantasy_stats_2019)

qb_stats_2023 = get_qb_stats(fantasy_stats_2023)
qb_stats_2022 = get_qb_stats(fantasy_stats_2022)
qb_stats_2021 = get_qb_stats(fantasy_stats_2021)
qb_stats_2020 = get_qb_stats(fantasy_stats_2020)
qb_stats_2019 = get_qb_stats(fantasy_stats_2019)

player_stats_2023 = pd.merge(player_stats_2023, qb_stats_2023, how='inner', left_on='Tm', right_on='Tm_QB')
player_stats_2022 = pd.merge(player_stats_2022, qb_stats_2022, how='inner', left_on='Tm', right_on='Tm_QB')
player_stats_2021 = pd.merge(player_stats_2021, qb_stats_2021, how='inner', left_on='Tm', right_on='Tm_QB')
player_stats_2020 = pd.merge(player_stats_2020, qb_stats_2020, how='inner', left_on='Tm', right_on='Tm_QB')
player_stats_2019 = pd.merge(player_stats_2019, qb_stats_2019, how='inner', left_on='Tm', right_on='Tm_QB')

ppr_2020 = player_stats_2020[['Player', 'PPR']]
ppr_2021 = player_stats_2021[['Player', 'PPR']]
ppr_2022 = player_stats_2022[['Player', 'PPR']]
ppr_2023 = player_stats_2023[['Player', 'PPR']]

player_stats_2019 = join_ppr_next_season(player_stats_2019, ppr_2020)
player_stats_2020 = join_ppr_next_season(player_stats_2020, ppr_2021)
player_stats_2021 = join_ppr_next_season(player_stats_2021, ppr_2022)
player_stats_2022 = join_ppr_next_season(player_stats_2022, ppr_2023)

"""## Team offense stats"""

offense_2023 = pd.read_csv('team_offense_2023.csv')
offense_2022 = pd.read_csv('team_offense_2022.csv')
offense_2021 = pd.read_csv('team_offense_2021.csv')
offense_2020 = pd.read_csv('team_offense_2020.csv')
offense_2019 = pd.read_csv('team_offense_2019.csv')


offense_2023_clean = clean_team_stats(offense_2023)
offense_2022_clean = clean_team_stats(offense_2022)
offense_2021_clean = clean_team_stats(offense_2021)
offense_2020_clean = clean_team_stats(offense_2020)
offense_2019_clean = clean_team_stats(offense_2019)

"""## College stats"""

college_stats_2022 = pd.read_csv('college_receiving_2022.csv')
college_stats_2021 = pd.read_csv('college_receiving_2021.csv')
college_stats_2020 = pd.read_csv('college_receiving_2020.csv')
college_stats_2019 = pd.read_csv('college_receiving_2019.csv')
college_stats_2018 = pd.read_csv('college_receiving_2018.csv')

player_college_stats_2022 = clean_college_stats(college_stats_2022)
player_college_stats_2021 = clean_college_stats(college_stats_2021)
player_college_stats_2020 = clean_college_stats(college_stats_2020)
player_college_stats_2019 = clean_college_stats(college_stats_2019)
player_college_stats_2018 = clean_college_stats(college_stats_2018)

"""## Combine tables"""

player_team_stats_2023 = pd.merge(player_stats_2023, offense_2023_clean, how='left', left_on='Tm', right_on='Abbreviation')
player_team_stats_2022 = pd.merge(player_stats_2022, offense_2022_clean, how='left', left_on='Tm', right_on='Abbreviation')
player_team_stats_2021 = pd.merge(player_stats_2021, offense_2021_clean, how='left', left_on='Tm', right_on='Abbreviation')
player_team_stats_2020 = pd.merge(player_stats_2020, offense_2020_clean, how='left', left_on='Tm', right_on='Abbreviation')
player_team_stats_2019 = pd.merge(player_stats_2019, offense_2019_clean, how='left', left_on='Tm', right_on='Abbreviation')

total_player_stats_2023 = pd.merge(player_team_stats_2023, player_college_stats_2022, how='left', on='Player')
total_player_stats_2022 = pd.merge(player_team_stats_2022, player_college_stats_2021, how='left', on='Player')
total_player_stats_2021 = pd.merge(player_team_stats_2021, player_college_stats_2020, how='left', on='Player')
total_player_stats_2020 = pd.merge(player_team_stats_2020, player_college_stats_2019, how='left', on='Player')
total_player_stats_2019 = pd.merge(player_team_stats_2019, player_college_stats_2018, how='left', on='Player')

"""## Upsample more recent years"""

total_player_stats_2022_copy1 = total_player_stats_2022.copy()
total_player_stats_2022_copy2 = total_player_stats_2022.copy()
total_player_stats_2022_copy3 = total_player_stats_2022.copy()
total_player_stats_2022_copy4 = total_player_stats_2022.copy()

total_player_stats_2021_copy1 = total_player_stats_2021.copy()
total_player_stats_2021_copy2 = total_player_stats_2021.copy()
total_player_stats_2021_copy3 = total_player_stats_2021.copy()

total_player_stats_2020_copy1 = total_player_stats_2020.copy()
total_player_stats_2020_copy2 = total_player_stats_2020.copy()

total_player_stats_2019_copy1 = total_player_stats_2019.copy()

total_player_stats_2022 = pd.concat([total_player_stats_2022, total_player_stats_2022_copy1])
total_player_stats_2022 = pd.concat([total_player_stats_2022, total_player_stats_2022_copy2])
total_player_stats_2022 = pd.concat([total_player_stats_2022, total_player_stats_2022_copy3])
total_player_stats_2022 = pd.concat([total_player_stats_2022, total_player_stats_2022_copy4])

total_player_stats_2021 = pd.concat([total_player_stats_2021, total_player_stats_2021_copy1])
total_player_stats_2021 = pd.concat([total_player_stats_2021, total_player_stats_2021_copy2])
total_player_stats_2021 = pd.concat([total_player_stats_2021, total_player_stats_2021_copy3])

total_player_stats_2020 = pd.concat([total_player_stats_2020, total_player_stats_2020_copy1])
total_player_stats_2020 = pd.concat([total_player_stats_2020, total_player_stats_2020_copy2])

total_player_stats_2019 = pd.concat([total_player_stats_2019, total_player_stats_2019_copy1])

"""## Final concat"""

total_stats = pd.concat([total_player_stats_2022, total_player_stats_2021])
total_stats = pd.concat([total_stats, total_player_stats_2020])
total_stats = pd.concat([total_stats, total_player_stats_2019])

total_stats.shape

"""### Cleanup before training"""

def final_cleaning(df):
  df['PPR'].fillna(0, inplace=True)
  df['G_college'].fillna(-1, inplace=True)
  df['Rec_college'].fillna(-1, inplace=True)
  df['Plays_college'].fillna(-1, inplace=True)
  df['Yds_college'].fillna(-1, inplace=True)
  df['Avg_college'].fillna(-1, inplace=True)
  df['TD_college'].fillna(-1, inplace=True)
  df['PF'].fillna(-1, inplace=True)
  df['Yds'].fillna(-1, inplace=True)
  df['Ply'].fillna(-1, inplace=True)
  df['TO'].fillna(-1, inplace=True)
  df['FL_y'].fillna(-1, inplace=True)
  df['rushing_attempts'].fillna(-1, inplace=True)
  df['rushing_yards'].fillna(-1, inplace=True)
  df['rushing_td'].fillna(-1, inplace=True)
  df['yards_per_rushing_attempt'].fillna(-1, inplace=True)
  df['rushing_1stdown'].fillna(-1, inplace=True)
  df['Pen'].fillna(-1, inplace=True)
  df['penalty_yards'].fillna(-1, inplace=True)
  df['1stPy'].fillna(-1, inplace=True)
  df['drives_score_pct'].fillna(-1, inplace=True)
  df['drives_to_pct'].fillna(-1, inplace=True)
  df['EXP'].fillna(-1, inplace=True)
  df['yards_per_play'].fillna(-1, inplace=True)
  df['1stD'].fillna(-1, inplace=True)

  df.drop(['Rk_x', 'Rk_y', 'Tm', 'FantPos', 'Abbreviation'], axis=1, inplace=True)
  return df

total_stats = final_cleaning(total_stats)
prediction_data_2024 = final_cleaning(total_player_stats_2023)

prediction_data_2024.drop_duplicates('Player', inplace=True)

"""## Create train and test sets"""

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

features = total_stats.drop('PPR', axis=1)

# Define the target variable as 'charges'
target = total_stats['PPR']

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                  # random_state=2023,
                                                  test_size=0.2)

# Display the shapes of the training and validation sets
X_train.shape, X_val.shape

"""## Scaling"""

validation_players = X_val['Player']
X_val.drop(['Player', 'Player_QB', 'Tm_QB', 'FantPos_QB'], axis=1, inplace=True)
X_train.drop(['Player', 'Player_QB', 'Tm_QB', 'FantPos_QB'], axis=1, inplace=True)

# Standardize Features

# Use StandardScaler to scale the training and validation data
scaler = StandardScaler()
#Fit the StandardScaler to the training data
scaler.fit(X_train)
# transform both the training and validation data
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

"""## Fit model and make predictions"""

# Import necessary libraries for calculating mean squared error and using the LightGBM regressor.
from sklearn.metrics import mean_squared_error as mse
from lightgbm import LGBMRegressor

# Create an instance of the LightGBM Regressor with the RMSE metric.
model = LGBMRegressor(metric='rmse')

# Train the model using the training data.
model.fit(X_train_scaled, Y_train)

# Make predictions on the training and validation data.
X_train['preds'] = model.predict(X_train_scaled)
X_val['preds'] = model.predict(X_val_scaled)

X_train['actual'] = Y_train

X_val['actual'] = Y_val

# Calculate and print the Root Mean Squared Error (RMSE) for training and validation predictions.
from sklearn.metrics import mean_squared_error as mse
import numpy as np
print("Training RMSE: ", np.sqrt(mse(X_train['actual'], X_train['preds'])))
print("Validation RMSE: ", np.sqrt(mse(X_val['actual'], X_val['preds'])))

X_val_reset = X_val.reset_index()
val_players_reset = validation_players.reset_index()

final_valid = pd.merge(X_val_reset, val_players_reset, left_index=True, right_index=True)

final_valid[['Player', 'preds', 'actual']].sort_values(by='preds', ascending=False)

"""## Predict 2024"""

validation_players_2024 = prediction_data_2024['Player']
prediction_data_2024.drop(['Player', 'PPR', 'Player_QB', 'Tm_QB', 'FantPos_QB'], axis=1, inplace=True)
scaled_2024_preds = scaler.transform(prediction_data_2024)

prediction_data_2024['preds'] = model.predict(scaled_2024_preds)

predicted_stats_2024 = pd.merge(prediction_data_2024, validation_players_2024, left_index=True, right_index=True)

model_preds = predicted_stats_2024[['Player', 'preds']].sort_values(by='preds', ascending=False).reset_index()

"""## Include Fantasypros predictions"""

import pandas as pd
# model_preds = pd.read_csv('top_wr_predicted.csv')
fpros_preds = pd.read_csv('../../../Downloads/FantasyPros_2024_Draft_WR_Rankings.csv')

fpros_preds.drop(['TIERS', 'TEAM', 'BYE WEEK', 'SOS SEASON', 'ECR VS. ADP'], axis=1, inplace=True)

model_preds['model_rank'] = model_preds.index+1
fpros_preds.rename(columns={'RK': 'Fpros_rk'}, inplace=True)

model_fpros_preds = pd.merge(model_preds, fpros_preds, how='inner', left_on='Player', right_on = 'PLAYER NAME')
model_fpros_preds.drop(['PLAYER NAME'], axis=1, inplace=True)

model_fpros_preds['comb_rank'] = (model_fpros_preds['model_rank'] + model_fpros_preds['Fpros_rk'])/2

model_fpros_preds = model_fpros_preds.sort_values('comb_rank', ascending=True)

model_preds

model_fpros_preds

model_fpros_preds.to_csv('top_wr_predicted_fpros_stats.csv', index=False)

"""## Volatility stats"""

import pandas as pd
weekly_ppr_2023 = pd.read_csv('../../../Downloads/WEEKLY_WR_PPR_2023.csv')
weekly_ppr_2022 = pd.read_csv('WEEKLY_WR_PPR_2022.csv')
weekly_ppr_2021 = pd.read_csv('WEEKLY_WR_PPR_2021.csv')
weekly_ppr_2020 = pd.read_csv('WEEKLY_WR_PPR_2020.csv')
weekly_ppr_2019 = pd.read_csv('WEEKLY_WR_PPR_2019.csv')

preds = pd.read_csv('top_wr_predicted.csv')

weekly_ppr_2019 = weekly_ppr_2019[['Player', 'STD_DEV']]
weekly_ppr_2020 = weekly_ppr_2020[['Player', 'STD_DEV']]
weekly_ppr_2021 = weekly_ppr_2021[['Player', 'STD_DEV']]
weekly_ppr_2022 = weekly_ppr_2022[['Player', 'STD_DEV']]
weekly_ppr_2023 = weekly_ppr_2023[['Player', 'STD_DEV']]


weekly_ppr_2019.rename(columns={'STD_DEV': 'STD_DEV_2019'}, inplace=True)
weekly_ppr_2020.rename(columns={'STD_DEV': 'STD_DEV_2020'}, inplace=True)
weekly_ppr_2021.rename(columns={'STD_DEV': 'STD_DEV_2021'}, inplace=True)
weekly_ppr_2022.rename(columns={'STD_DEV': 'STD_DEV_2022'}, inplace=True)
weekly_ppr_2023.rename(columns={'STD_DEV': 'STD_DEV_2023'}, inplace=True)

preds = pd.merge(preds, weekly_ppr_2019, how='left', on='Player')
preds = pd.merge(preds, weekly_ppr_2020, how='left', on='Player')
preds = pd.merge(preds, weekly_ppr_2021, how='left', on='Player')
preds = pd.merge(preds, weekly_ppr_2022, how='left', on='Player')
preds = pd.merge(preds, weekly_ppr_2023, how='left', on='Player')

preds.to_csv('preds_w_std_dev.csv', index=False)

