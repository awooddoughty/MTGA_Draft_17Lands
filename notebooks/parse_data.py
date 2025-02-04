from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import re
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessPool as Pool
import warnings
from functools import partial

import dill
dill.settings['recurse'] = True

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
pool = Pool(8)

INFO_COLS = ['draft_id', 'rank', 'event_match_wins', 'pack_number', 'pick_number', 'pick_maindeck_rate', 'user_n_games_bucket', 'user_game_win_rate_bucket']
METRICS = ['alsa_normalized', 'alsa_normalized_squared', 'wr_normalized']
MAIN_METRIC = 'alsa_normalized_squared'
COLORS = ['B', 'W', 'U', 'G', 'R']


class SetParser:
    def __init__(self, set_name, draft_results_file, card_values_file, num_cards_in_pack):
        self.set_name = set_name
        self.filepath_stub = f'/Users/awooddoughty/Documents/mtg/{set_name}'
        self.draft_results_file = draft_results_file
        self.num_cards_in_pack = num_cards_in_pack
        self.card_values = pd.read_csv(card_values_file)
        self.add_metrics()
        self.row_parser = RowParser(card_values=self.card_values)
        self.decision_stats = DecisionStats()

    def add_metrics(self):
        self.card_values['alsa_normalized'] = pd.Series(1 - MinMaxScaler().fit_transform(self.card_values['ALSA'].values.reshape(-1, 1)).squeeze())
        self.card_values['alsa_normalized_squared'] = self.card_values['alsa_normalized']**2
        self.card_values['gih_wr'] = self.card_values['GIH WR'].str.extract(r'(.*)%').astype(float)
        self.card_values['gih_wr'] = self.card_values['gih_wr'].fillna(self.card_values['gih_wr'].min())
        self.card_values['wr_normalized'] = pd.Series(StandardScaler().fit_transform(self.card_values['gih_wr'].values.reshape(-1, 1)).squeeze())
        self.card_values['Color'] = self.card_values['Color'].fillna('A')

    def row_parse(self, df):
        df['num_rows'] = df.groupby('draft_id')['draft_id'].transform(lambda x: x.count())
        df['completed'] = np.where((df['event_match_losses'] == 3) | (df['event_match_wins'] == 7), 1, 0)
        df = df.loc[lambda x: (x['num_rows'] == 3 * self.num_cards_in_pack) & (x['completed'] == 1)]
        df['pick_number'] = df['pick_number'] + 1
        df['pack_number'] = df['pack_number'] + 1

        data = pd.DataFrame(
            pool.map(self.row_parser.compute_stats, [row for _, row in df.iterrows()])
        )

        pack_cols = [col for col in data.columns if col.startswith('current_pack')]
        data = data.join(
            data.loc[
                lambda x: x['pick_number'] > 1
            ].groupby(['draft_id', 'pack_number'], sort=False)[pack_cols].apply(
                lambda x: x.fillna(0).cumsum()
            ).rename(columns=lambda x: re.sub('current_pack', 'open_lane', x)).reset_index(drop=True, level=[0, 1])
        )

        data = data.merge(
            self.decision_stats.compute_pool_and_deck_values(data=data),
            on='draft_id',
        )
        return data
    
    def gen_decision_stats(self, data):
        return pd.DataFrame(
            [
                r
                for rows in pool.map(
                    partial(self.decision_stats.extract_options, metric=MAIN_METRIC),
                    [row for _, row in data.iterrows()]
                )
                for r in rows
            ]
        )
    
    def parse_set(self, chunksize):
        Path(f'{self.filepath_stub}/row_data').mkdir(parents=True, exist_ok=True)
        Path(f'{self.filepath_stub}/final_data').mkdir(parents=True, exist_ok=True)

        with pd.read_csv(self.draft_results_file, chunksize=chunksize) as reader:
            for i, df in enumerate(reader):
                data = self.row_parse(df=df)
                data.to_parquet(f'{self.filepath_stub}/row_data/{i}.parquet')
                print(f'Saved data: {i}')
                final_data = self.gen_decision_stats(data=data)
                final_data.to_parquet(f'{self.filepath_stub}/final_data/{i}.parquet')
                print(f'Saved final data: {i}')


class RowParser:
    def __init__(self, card_values):
        self.card_values = card_values

    def _get_card_values(self, row, col_type):
        relevant_data = row.loc[
            lambda y: y.index.str.startswith(col_type)
        ].loc[
            lambda y: y > 0
        ]
        relevant_data.index = relevant_data.index.str.extract(rf'{col_type}_(.*)', expand=False)
        all_cards = [
            self.card_values.loc[lambda x: x['Name'] == name]
            for name, num in relevant_data.items()
            for _ in range(num)  
        ]
        if len(all_cards) > 0:
            return pd.concat(all_cards)
        else:
            return pd.DataFrame(columns=['Color'] + METRICS)
    
    @staticmethod
    def _split_multicolor(cards):
        cards['num_colors'] = cards['Color'].apply(len)
        singlecolor_cards = cards.loc[lambda x: x['num_colors'] == 1]
        multicolor_cards = cards.loc[lambda x: x['num_colors'] > 1]
        multicolor_cards['Color'] = multicolor_cards['Color'].apply(list)
        multicolor_cards = multicolor_cards.explode(column='Color')
        for metric in METRICS:
            multicolor_cards[metric] /= multicolor_cards['num_colors']
        cards = pd.concat((singlecolor_cards, multicolor_cards))
        return cards
    
    def parse_pick(self, row):
        cards = self._get_card_values(row=row, col_type='pack_card')
        cards = self._split_multicolor(cards=cards)
        return cards.groupby('Color')[METRICS].max()
    
    def parse_pool(self, row):
        cards = self._get_card_values(row=row, col_type='pool')
        cards = self._split_multicolor(cards=cards)
        return cards.groupby('Color')[METRICS].sum()
    
    def compute_stats(self, row):
        pool_stats = {
            'current_pool_' + '_'.join(k): v
            for k, v in self.parse_pool(row).stack().to_dict().items()
        }
        pack_stats = {
            'current_pack_' + '_'.join(k): v
            for k, v in self.parse_pick(row).stack().to_dict().items()
        }
        actual_pick = self.card_values.loc[
            lambda x: x['Name'] == row['pick']
        ][['Color', 'Rarity'] + METRICS].add_prefix('actual_').T.squeeze()
        return {
            **row[INFO_COLS].to_dict(),
            **pool_stats,
            **pack_stats,
            **actual_pick.to_dict(),
        }
    
class DecisionStats:
    def __init__(self):
        pass
    
    @staticmethod
    def compute_pool_and_deck_values(data):
        return pd.concat(
            [
                data.groupby('draft_id').apply(
                    lambda x: (x['pick_maindeck_rate'] * x[f'actual_{metric}']).sum()
                ).rename(f'deck_{metric}')
                for metric in METRICS
            ] + [
                data.groupby('draft_id')[
                    f'actual_{metric}'
                ].sum().rename(f'final_pool_{metric}')
                for metric in METRICS
            ],
            axis=1
        )
    
    @staticmethod
    def compute_current_colors(row, metric):
        pool_cols = [
            col for col in row.index
            if re.match(rf'current_pool_[{"".join(COLORS)}]_{metric}$', col)
        ]
        return row[pool_cols].dropna().sort_values(
            ascending=False
        ).iloc[:2].index.str.extract(
            rf'current_pool_(\w)_{metric}', expand=False
        ).tolist()

    def extract_options(self, row, metric):
        rows = []
        current_colors = self.compute_current_colors(row=row, metric=metric)
        if len(current_colors) == 2:
            best_option_in_current_colors = row.loc[
                lambda x: x.index.str.match(rf'current_pack_[{"".join(current_colors)}]_{metric}')
            ].max()
            current_pool_second_color = row[f'current_pool_{current_colors[-1]}_{metric}']
            open_lane_second_color = row[f'open_lane_{current_colors[-1]}_{metric}']
            actual_color = row['actual_Color']
            for color in [c for c in COLORS if c not in current_colors]:
                best_option_in_other = row[f'current_pack_{color}_{metric}']
                open_lane_in_other = row[f'open_lane_{color}_{metric}']
                if ~np.isnan(best_option_in_other):
                    rows.append({
                        **row[
                            [
                                'draft_id',
                                'rank',
                                'event_match_wins',
                                'pack_number',
                                'pick_number',
                                'pick_maindeck_rate',
                                'user_n_games_bucket',
                                'user_game_win_rate_bucket',
                            ] + [
                                f'final_pool_{metric}' for metric in METRICS
                            ] + [
                                f'deck_{metric}' for metric in METRICS
                            ]
                        ].to_dict(),
                        'best_option_in_colors': best_option_in_current_colors,
                        'current_pool_second_color': current_pool_second_color,
                        'open_lane_second_color': open_lane_second_color,
                        'best_option_in_other': best_option_in_other,
                        'open_lane_in_other': open_lane_in_other,
                        'other_color': color,
                        'switch': int(actual_color == color),
                        'delta_card_value': best_option_in_other - best_option_in_current_colors,
                        'delta_open_lane': open_lane_in_other - open_lane_second_color,
                    })
        return rows