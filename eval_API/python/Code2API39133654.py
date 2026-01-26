import matplotlib.pyplot as plt

def create_subplots_with_titles(subplot_titles, layout=(2,2)):
    fig = plt.figure()
    axes = []
    total_subplots = layout[0] * layout[1]
    
    for i in range(total_subplots):
        ax = fig.add_subplot(layout[0]*100 + layout[1]*10 + (i+1))
        if i < len(subplot_titles):
            ax.title.set_text(subplot_titles[i])
        axes.append(ax)
    
    plt.show()
    return fig
