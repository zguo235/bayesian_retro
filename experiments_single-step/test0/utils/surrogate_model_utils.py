import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


def test_model(model, test_dataset, loss_func, title, fig_path=None, target_idx=None):
    test_X, test_y = test_dataset
    test_pred = model.predict(test_X)
    test_true = test_y
    loss = loss_func(test_true, test_pred)
    print(title, '\nTest loss:', loss)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(test_true, test_pred, c='blue', s=10,
               label='{}: {:.5f}'.format(loss_func.__name__, loss))
    if target_idx is not None:
        ax.scatter(test_true[target_idx], test_pred[target_idx], c='red', s=50)
    ax.set_title(title)
    lim = np.max([test_pred.max(), test_true.max()])
    ax.set_xlim(-0.5, lim * 1.05)
    ax.set_ylim(-0.5, lim * 1.05)
    ax.set_xlabel('Observation')
    ax.set_ylabel('Prediction')
    ax.legend(loc='lower right')
    if fig_path is None:
        plt.show()
    else:
        fig.savefig(fig_path)
        plt.close(fig)
