import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

def make_chart(dataframe, axis, figure,title,colors):
    color1=colors[0]
    color2=colors[1]
    df = dataframe
    array_values = df[df.columns[0]].to_list()
    print(array_values)
    df2 = np.array([df.columns[0]])
    df2 = np.concatenate([df2,array_values])
    print(df2)
    indices = []
    failures=[]
    times=[]
    means=[]
    stds=[]
    for i in df2:
        x = i.split("\t")
        indices.append(x[0])
        failures.append(1-float(x[1])/200)
        times.append([float(i) for i in x[2:]])
        means.append(np.mean([float(i) for i in x[2:]]))
        stds.append(np.std([float(i) for i in x[2:]]))

    df3 = pd.DataFrame(failures, index=indices, columns=['prob of success']) 
    df3['times'] = times
    df3['means'] = means
    df3['std. devs'] = stds
    df3.index.name = 'index1'

    df3 = df3.sort_index()

    df3_indices = df3.index.unique()

    probs_of_success=[]
    probs_of_sucess_std = []
    mean_times=[]
    mean_times_std=[]
    for i in df3_indices:
        object = df3.loc[i]
        probs_of_success.append(np.mean(object['prob of success'].to_list()))
        probs_of_sucess_std.append(np.std(object['prob of success'].to_list()))
        mean_times.append(np.mean(object['means'].to_list()))
        mean_times_std.append(np.std(object['means'].to_list()))

    df4 = pd.DataFrame(probs_of_success, index = df3_indices, columns=['mean probs of success'])
    df4['std dev success'] = probs_of_sucess_std
    df4['mean time'] = mean_times
    df4['std dev mean time'] = mean_times_std
    print(df4)

    # create figure and axis objects with subplots()
    fig=figure
    ax = axis
    # make a plot
    # ax.plot(df3.index.unique(), df4['mean probs of success'], color="red", marker="o")
    ax.errorbar(df3.index.unique(), df4['mean probs of success'], capsize=10,color=color1, marker="o", yerr=df4['std dev success'])
    # set x-axis label
    ax.set_xlabel("Velocity",fontsize=14)
    # set y-axis label
    ax.set_ylabel("Prob of Success",color=color1,fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax3=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax3.errorbar(df3.index.unique(), df4['mean time'],color=color2,capsize=10,marker="o",yerr=df4['std dev mean time'])
    ax3.set_ylabel("Time to find Source",color=color2,fontsize=14)
    ax.set_xticks(df3.index.unique())
    labels = []
    for tick in ax.get_xticks():
        label = f'{tick:0<5.1f}'
        labels.append(label[:-1].rstrip('.'))

    ax.set_xticklabels(labels)
    ax.set_title(title)

if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(1, 2)
    df0 = pd.read_csv('paula_1000_seeds.txt')
    df = pd.read_csv("greedy_algorithm_D_2_R_1_talia.dat")
    length = len(df0.index)
    df = df.loc[0:length]
    make_chart(df0, axis=ax1, figure=fig, title='Rate=2, Diffusion=2', colors=['green','orange'])
    make_chart(df, axis=ax2, figure=fig, title='Rate=1, Diffusion=2', colors=['red','blue'])
    plt.tight_layout()
    plt.show()

