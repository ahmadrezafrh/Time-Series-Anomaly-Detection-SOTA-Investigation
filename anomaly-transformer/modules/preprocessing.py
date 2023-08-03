# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import VarianceThreshold

def seaborn_cm(cm, ax, tick_labels, fontsize=14, title=None, sum_actual="over_columns",
               xrotation=0, yrotation=0):
    """
    Function to plot a confusion matrix
    """
    from matplotlib import cm as plt_cmap
    group_counts = ["{:0.0f}".format(value) for value in cm.flatten()]
    if sum_actual == "over_columns":
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif sum_actual == "over_rows":
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
    else:
        print("sum_actual must be over_columns or over_rows")
        exit()
    cm = np.nan_to_num(cm)
    mean_acc = np.mean(np.diag(cm)[cm.sum(axis=1) != 0])
    std_acc = np.std(np.diag(cm))
    group_percentages = ["{:0.0f}".format(value*100) for value in cm.flatten()]
    cm_labels = [f"{c}\n{p}%" for c, p in zip(group_counts, group_percentages)]
    cm_labels = np.asarray(cm_labels).reshape(len(tick_labels), len(tick_labels))
    sns.heatmap(cm,
                ax=ax,
                annot=cm_labels,
                fmt='',
                cbar=False,
                cmap=plt_cmap.Greys,
                linewidths=1, linecolor='black',
                annot_kws={"fontsize": fontsize},
                xticklabels=tick_labels,
                yticklabels=tick_labels)
    ax.set_yticklabels(ax.get_yticklabels(), size=fontsize, rotation=yrotation)
    ax.set_xticklabels(ax.get_xticklabels(), size=fontsize, rotation=xrotation)
    if title:
        title = f"{title}\nMean accuracy {mean_acc * 100:.1f} +- {std_acc * 100:.1f}"
    else:
        title = f"Mean accuracy {mean_acc * 100:.1f} +- {std_acc * 100:.1f}"
    ax.set_title(title)
    if sum_actual == "over_columns":
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    else:
        ax.set_ylabel("Predicted")
        ax.set_xlabel("Actual")
    ax.axis("off")
    
    
    
def get_df_action(filepaths_csv, filepaths_meta, action2int=None, delimiter=";"):
    print("Loading data.")
    dfs_meta = list()
    for filepath in filepaths_meta:
        df_m = pd.read_csv(filepath, sep=delimiter)
        df_m.str_repr = df_m.str_repr.str.replace('True', 'true')
        df_m['filepath'] = filepath
        dfs_meta.append(df_m)

    df_meta = pd.concat(dfs_meta)
    df_meta.index = pd.to_datetime(df_meta.init_timestamp.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    df_meta['completed_timestamp'] = pd.to_datetime(df_meta.completed_timestamp.astype('datetime64[ms]'),
                                                    format="%Y-%m-%dT%H:%M:%S.%f")
    df_meta['init_timestamp'] = pd.to_datetime(df_meta.init_timestamp.astype('datetime64[ms]'),
                                               format="%Y-%m-%dT%H:%M:%S.%f")



    actions = df_meta.str_repr.unique()
    dfs = [pd.read_csv(filepath_csv, sep=";") for filepath_csv in filepaths_csv]
    df = pd.concat(dfs)
    df = df.sort_index(axis=1)
    df.index = pd.to_datetime(df.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    columns_to_drop = [column for column in df.columns if "Abb" in column or "Temperature" in column]
    df.drop(["machine_nameKuka Robot_export_active_energy",
             "machine_nameKuka Robot_import_reactive_energy"] + columns_to_drop, axis=1, inplace=True)


    df_action = list()
    for action in actions:
        for index, row in df_meta[df_meta.str_repr == action].iterrows():
            start = row['init_timestamp']
            end = row['completed_timestamp']
            df_tmp = df.loc[start: end].copy()
            df_tmp['action'] = action
            df_tmp['duration'] = str((row['completed_timestamp'] - row['init_timestamp']).total_seconds())
            df_action.append(df_tmp)
    df_action = pd.concat(df_action, ignore_index=True)
    df_action.index = pd.to_datetime(df_action.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    df_action = df_action[~df_action.index.duplicated(keep='first')]
    df = df.dropna(axis=0)
    df_action = df_action.dropna(axis=0)

    if action2int is None:
        action2int = dict()
        j = 1
        for label in df_action.action.unique():
            action2int[label] = j
            j += 1

    df_merged = df.merge(df_action[['action']], left_index=True, right_index=True, how="left")
    df_idle = df_merged[df_merged['action'].isna()].copy()
    df_idle['action'] = 'idle'
    df_idle['duration'] = df_action.duration.values.astype(float).mean().astype(str)
    df_action = pd.concat([df_action, df_idle])
    action2int['idle'] = 0
    print(f"Found {len(set(df_action['action']))} different actions.")
    print("Loading data done.\n")

    return df_action, df, df_meta, action2int


class load_data:
    
    def __init__(self, freq=10, rand_size=None, select_signals=None, select_action=None, save=True):
        self.freq = freq
        self.rand_size = rand_size
        self.select_signals = select_signals
        self.select_action = select_action
        self.save = save
        
    def train(self, path):
        filepaths_csv = [os.path.join(path, f"rec{r}_20220811_rbtc_{1/self.freq}s.csv") for r in [0, 2, 3, 4]]
        filepaths_meta = [os.path.join(path, f"rec{r}_20220811_rbtc_{1/self.freq}s.metadata") for r in [0, 2, 3, 4]]
        df_action, df, df_meta, action2int = get_df_action(filepaths_csv, filepaths_meta)
        df_action.index = df_action.index.map(mapper=(lambda x: pd.Timestamp(x).timestamp()))
        df_action = df_action.drop(['time'], axis=1)
        df_action.reset_index(inplace=True)
        df_action = df_action.rename(columns = {'time':'timestamp'})
        
        if self.select_action:
            df_action = df_action[df_action['action']==self.select_action]
            df_action = df_action.drop(['action'], axis=1)
        else:
            df_action['action'] = df_action['action'].map(action2int, na_action='ignore')
            
        train_data = df_action.drop(['timestamp'], axis=1)
        if self.select_signals:
            train_data = train_data[self.select_signals]
        
        self.sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        train_data = self.sel.fit_transform(train_data)
        
        
        if self.rand_size:
            self.idx = np.random.randint(train_data.shape[1], size=self.rand_size)
            train_data = train_data[:,self.idx]
        
        if self.save:
            print(f'Train data shape: {train_data.shape}')
            np.save('./dataset/train.npy', train_data)
        
        return train_data, df


    def test(self, data_path):
        
        collisions = pd.read_excel(os.path.join(data_path, "20220811_collisions_timestamp.xlsx"))
        collisions.Timestamp = collisions.Timestamp - pd.to_timedelta([2] * len(collisions.Timestamp), 'h')

        collisions = collisions.rename(columns = {'Timestamp':'timestamp'})
        collisions['timestamp'] = collisions['timestamp'].map(lambda x: pd.Timestamp(x).timestamp())
        
        filepath_csv = [os.path.join(data_path, f"rec{r}_collision_20220811_rbtc_{1/self.freq}s.csv") for r in [1, 5]]
        filepath_meta = [os.path.join(data_path, f"rec{r}_collision_20220811_rbtc_{1/self.freq}s.metadata") for r in [1, 5]]
        df_action, df, df_meta, action2int = get_df_action(filepath_csv, filepath_meta)
        df_action.index = df_action.index.map(mapper=(lambda x: pd.Timestamp(x).timestamp()))
        
        df_action = df_action.drop(['time'], axis=1)
        df_action.reset_index(inplace=True)
        df_action = df_action.rename(columns = {'time':'timestamp'})
        if self.select_action:
            df_action = df_action[df_action['action']==self.select_action]
            df_action = df_action.drop(['action'], axis=1)
        else:
            df_action['action'] = df_action['action'].map(action2int, na_action='ignore')
        test_data = df_action.copy()
        
        tuples = []
        for c, col in enumerate(collisions['Inizio/fine']):
            if col=='i':
                tuples.append((collisions['timestamp'][c],collisions['timestamp'][c+1]))
        
        labels = []
        for sample in test_data.timestamp:
            d=0
            for tup in tuples:
                
                if sample>=tup[0] and sample<=tup[1]:
                    k=1
                    break
                else:
                    k=0
                    d+=1
            labels.append(k)
        labels = np.array(labels)
        
        test_data = test_data.drop(['timestamp'], axis=1)
        if self.select_signals:
            test_data = test_data[self.select_signals]
        test_data = self.sel.transform(test_data)
        if self.rand_size:
            test_data = test_data[:,self.idx]
        
        if self.save:
            print(f'Test data shape: {test_data.shape}')
            np.save('./dataset/test.npy', test_data)
            print(f'Test Labels shape: {labels.shape}')
            np.save('./dataset/test_label.npy', labels)
        
        return test_data, labels, df
    
    





def plot_data(path, freq, signals, file_name='data'):
    plt.style.use("Solarize_Light2")
    filepaths_csv = [os.path.join(path, f"rec{r}_20220811_rbtc_{1/freq}s.csv") for r in [0, 2, 3, 4]]
    filepaths_meta = [os.path.join(path, f"rec{r}_20220811_rbtc_{1/freq}s.metadata") for r in [0, 2, 3, 4]]
    df_action, df, df_meta, action2int = get_df_action(filepaths_csv, filepaths_meta)
    fig = go.Figure()
    start = df.index[1000]
    df_reduced = df.loc[start:]
    duration = 120
    time_delta = df_reduced.index - start 
    df_interval = df_reduced[time_delta.total_seconds() <= duration]
    j = 0
    
    n_colors = len(signals)
    colors = px.colors.sample_colorscale("greys", [n/(n_colors -1) for n in range(n_colors)])  # From continuous colormap
    colors = px.colors.qualitative.Set2  # From discrete colormap, see https://plotly.com/python/discrete-color/
    df_signals = df_interval[signals].select_dtypes(['number'])
    df_signals = df_signals / df_signals.max()
    fig = px.line(df_signals, x=df_signals.index, y=df_signals.columns, color_discrete_sequence=colors)
    
    colors_action = px.colors.qualitative.Antique
    j = 0
    for action in df_action.loc[df_interval.index].action.unique():
        df_action_interval = df_action.loc[df_interval.index]
        df_action_single_action = df_action_interval[df_action_interval['action'] == action]
        fig.add_trace(go.Scatter(
            x=df_action_single_action.index,
            y=[-0.3] * len(df_action_single_action.index),
            line_shape="hv",
            line=dict(color=colors_action[j], width=2.5),
            name=action))
        j += 1
        
    fig.update_layout(
    title="Some signals",
    xaxis_title="Time",
    yaxis_title="",
    legend_title="Legend",
    font=dict(
        family="Courier New, monospace",
        size=12,
        color="Black"
        )
    )
    fig.show()
    plt.savefig(f"figs/{file_name}_1.png",bbox_inches='tight')
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    df_signals.plot(ax=ax)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.savefig(f"figs/{file_name}_2.png",bbox_inches='tight')