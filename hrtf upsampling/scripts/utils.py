import matplotlib.pyplot as plt
import os


def plot_loss(config,history,fold):
    plt.clf()
    plt.plot(history['Train Loss'][5:],label = 'Train Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history['Valid Loss'][5:],label = 'Valid Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()
    if not os.path.exists(config.loss_png_pth):
        os.makedirs(config.loss_png_pth)
    plt.savefig(config.loss_png_pth +'Loss_'+str(fold)+'.png')
    print(min(history['Train Loss']))
    print(min(history['Valid Loss']))
    best_model_ind = history['Valid Loss'].index(min(history['Valid Loss']))
    print(best_model_ind)
    print(history['Train Loss'][best_model_ind])
    print(history['Valid Loss'][best_model_ind])
    return best_model_ind
