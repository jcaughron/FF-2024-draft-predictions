import pandas as pd
from utils.wr_data_prep_funcs import clean_before_training, clean_team_stats, clean_college_stats, clean_player_stats, get_qb_stats, join_ppr_next_season
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error as mse
import numpy as np

class FantasyPredictions:
    @staticmethod
    def prep_training_data():
        fantasy_stats_2023 = pd.read_csv('data/Fantasy_player_stats_2023.csv')
        fantasy_stats_2022 = pd.read_csv('data/Fantasy_player_stats_2022.csv')
        fantasy_stats_2021 = pd.read_csv('data/Fantasy_player_stats_2021.csv')
        fantasy_stats_2020 = pd.read_csv('data/Fantasy_player_stats_2020.csv')
        fantasy_stats_2019 = pd.read_csv('data/Fantasy_player_stats_2019.csv')

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

        player_team_stats_2023 = pd.merge(player_stats_2023, offense_2023_clean, how='left', left_on='Tm',
                                          right_on='Abbreviation')
        player_team_stats_2022 = pd.merge(player_stats_2022, offense_2022_clean, how='left', left_on='Tm',
                                          right_on='Abbreviation')
        player_team_stats_2021 = pd.merge(player_stats_2021, offense_2021_clean, how='left', left_on='Tm',
                                          right_on='Abbreviation')
        player_team_stats_2020 = pd.merge(player_stats_2020, offense_2020_clean, how='left', left_on='Tm',
                                          right_on='Abbreviation')
        player_team_stats_2019 = pd.merge(player_stats_2019, offense_2019_clean, how='left', left_on='Tm',
                                          right_on='Abbreviation')

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
        total_stats = clean_before_training(total_stats)
        prediction_data_2024 = clean_before_training(total_player_stats_2023)

        prediction_data_2024.drop_duplicates('Player', inplace=True)
        print(total_stats.shape)

        return total_stats

    @staticmethod
    def model_train_validate(df):
        features = df.drop('PPR', axis=1)

        # Define the target variable as 'charges'
        target = df['PPR']

        # Split the data into training and validation sets
        X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                          test_size=0.2)

        # Display the shapes of the training and validation sets
        print(X_train.shape, X_val.shape)

        """## Scaling"""

        validation_players = X_val['Player']
        X_val.drop(['Player', 'Player_QB', 'Tm_QB', 'FantPos_QB'], axis=1, inplace=True)
        X_train.drop(['Player', 'Player_QB', 'Tm_QB', 'FantPos_QB'], axis=1, inplace=True)

        # Standardize Features

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LGBMRegressor(metric='rmse')

        # Train the model using the training data.
        model.fit(X_train_scaled, Y_train)

        # Make predictions on the training and validation data.
        X_train['preds'] = model.predict(X_train_scaled)
        X_val['preds'] = model.predict(X_val_scaled)

        X_train['actual'] = Y_train

        X_val['actual'] = Y_val

        # Calculate and print the Root Mean Squared Error (RMSE) for training and validation predictions.
        print("Training RMSE: ", np.sqrt(mse(X_train['actual'], X_train['preds'])))
        print("Validation RMSE: ", np.sqrt(mse(X_val['actual'], X_val['preds'])))
        return model


