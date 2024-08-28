import pandas as pd

def clean_player_stats(df):
  df = df.drop(['DKPt',
 'FDPt',
 'PosRank',
 'OvRank',
 '-9999', 'FantPt', '2PM', '2PP', 'TD.3'], axis=1)

  df = df.rename(columns={"TD": "TD_passing", "Yds": "Yds_passing", "Att": "Att_passing", "Att.1": "Att_rushing", "Yds.1": "Yds_rushing", "TD.1": "TD_rushing"})


  player_stats = df[df['FantPos'].isin(['WR'])]
  player_stats['Player'] = player_stats['Player'].str.replace(r'\*.*', '', regex=True)
  player_stats = player_stats.fillna(0)
  return player_stats

def get_qb_stats(df):
  df = df.drop(['DKPt',
 'FDPt',
 'PosRank',
 'OvRank',
 '-9999', 'FantPt', '2PM', '2PP', 'TD.3'], axis=1)

  df = df.rename(columns={"TD": "TD_passing", "Yds": "Yds_passing", "Att": "Att_passing", "Att.1": "Att_rushing", "Yds.1": "Yds_rushing", "TD.1": "TD_rushing"})


  player_stats = df[df['FantPos'].isin(['QB'])]
  player_stats['Player'] = player_stats['Player'].str.replace(r'\*.*', '', regex=True)
  player_stats = player_stats.fillna(0)
  player_stats = player_stats.add_suffix(f"_QB", axis=1)
  return player_stats

def clean_team_stats(df):
  team_name_conversion = pd.read_csv('team_name_conversion.csv')
  df = pd.merge(df, team_name_conversion, how='inner', left_on='Tm', right_on = 'Name')
# 'Cmp', 'Att', 'Yds.1', '1stD.1', 'TD', 'Int', 'NY/A',
  df.drop(['Tm', 'Name', 'Conference', 'Division', 'G', 'ID'], axis=1, inplace=True)
  df.rename(columns={'Att.1': 'rushing_attempts',
                                'Yds.2': 'rushing_yards',  'TD.1': 'rushing_td',  '1stD.2': 'rushing_1stdown',
                                'Yds.3': 'penalty_yards',  'Y/P': 'yards_per_play',
                                'Y/A': 'yards_per_rushing_attempt',  'Sc%': 'drives_score_pct',  'TO%': 'drives_to_pct',}, inplace=True)
  return df




def clean_college_stats(df):
  df.loc[df['Conf'].isin(['SEC', 'Big 12', 'ACC', 'Big Ten', 'Pac-12']), 'power5'] = 1
  df['power5'].fillna(0, inplace=True)
  df.drop(['-9999', 'School', 'Att', 'Yds', 'Avg', 'TD', 'Att', 'Yds.1', 'TD.1', 'Avg.1', 'Rk', 'Conf'], axis=1, inplace=True)
  df.rename(columns={'Yds.2': 'Yds', 'Avg.2': 'Avg', 'TD.2': 'TD'}, inplace=True)
  df['Player'] = df['Player'].str.replace(r'\*.*', '', regex=True)
  df  = df.add_suffix(f'_college', axis=1)
  df.rename(columns={f'Player_college': 'Player'}, inplace=True)
  return df

def join_ppr_next_season(df, ppr_df):
  df.drop('PPR', axis=1, inplace=True)
  df = pd.merge(df, ppr_df, how='left', on='Player')
  return df

def clean_before_training(df):
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