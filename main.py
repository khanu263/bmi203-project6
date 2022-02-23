from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():

    # define features to use
    f = ['Penicillin V Potassium 500 MG',
         'Computed tomography of chest and abdomen',
         'Plain chest X-ray (procedure)',
         'Low Density Lipoprotein Cholesterol',
         'Creatinine',
         'AGE_DIAGNOSIS']

    # load data with default settings
    X_train, X_val, y_train, y_val = utils.loadDataset(features = f, split_percent = 0.8, split_state = 42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    # perform regression
    log_model = logreg.LogisticRegression(num_feats = 6, learning_rate = 1e-5, tol = 1e-6)
    log_model.train_model(X_train, y_train, X_val, y_val)
    log_model.plot_loss_history()
    plt.show()

if __name__ == "__main__":
    main()
