import qml
from glob import glob
import pdb
from xyz2mol import *
from qml.utils import alchemy
import pandas as pd
from rdkit import Chem
import ast
from drfp import DrfpEncoder
from gzip_regressor import regress, cross_val_and_fit_kernel_ridge, predict_kernel_ridge_regression
from smiles_tokenizer import tokenize
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge

random.seed(42)

def xyz2smiles(path):
    atoms, charge, coordinates = read_xyz_file(path)
    return atoms, charge, coordinates

def side_2_smiles(PATHS):
    SMILES = []
    for path in PATHS:
        
        atoms, charge, coordinates = xyz2smiles(path)
        print(atoms)
        if len(atoms) == 1:
            atoms = [alchemy.ELEMENT_NAME[atoms[0]]]
            #coordinates = [coordinates]
            SMILES.append(atoms[0])

        else:
            mol = xyz2mol(atoms, coordinates)[0]
            SMILES.append(Chem.MolToSmiles(mol))

    return SMILES


def rxn_smiles(reac_SMILES,prod_SMILES):
    left  = "{}.{}>>".format(reac_SMILES[0],reac_SMILES[1])
    right = "{}.{}".format(prod_SMILES[0],prod_SMILES[1])
    return left + right

def learning_curve(X_train, X_test, y_train, y_test):

    #gammas, lambdas =  np.logspace(-3, 3, 7), [1e-8]
    gammas, lambdas =  np.logspace(-1, 4, 15), [1e-8, 1e-6]

    errors = []
    for n in N_train:
        best_alpha, best_gamma, best_lambda_, best_score =  cross_val_and_fit_kernel_ridge(X_train[:n], y_train[:n],5, gammas, lambdas)
        print("n = {}, best_gamma = {}, best_lambda = {}, best_score = {}".format(n, best_gamma, best_lambda_, best_score))
        test_preds = predict_kernel_ridge_regression(X_train[:n], X_test, best_alpha, best_gamma)
        # compute MAE
        mae = np.mean(np.abs(test_preds - y_test))
        #print n and mae
        print("n = {}, MAE = {}".format(n, mae))
        errors.append(mae)
    
    return errors

#read pandas dataframe

if __name__ == "__main__":

    PREPROCESS,REGRESSION = False,True
    PLOT = True
    if PREPROCESS:
        reactions_SN2 = pd.read_csv("/home/jan/projects/testing_zip/molzip_react/data/SN2-20/reactions.csv")
        REACTANTS, PRODUCTS, Y = reactions_SN2["reactant"].values, reactions_SN2["product"].values, reactions_SN2["rxn_nrj"].values
        REACT_SMILES = []
        for reacts, products, y in zip(REACTANTS, PRODUCTS, Y):
            reacts, products = ast.literal_eval(reacts), ast.literal_eval(products)
            reactant_SMILES,products_SMILES =  side_2_smiles(reacts), side_2_smiles(products)
            reac_smi = rxn_smiles(reactant_SMILES,products_SMILES)
            print(reac_smi)
            REACT_SMILES.append(reac_smi)
            #pdb.set_trace()

        REACT_SMILES = np.array(REACT_SMILES)
        fps, mapping = DrfpEncoder.encode(REACT_SMILES, mapping=True, n_folded_length=512)
        
        np.savez_compressed("react_SN2-20.npz", fps=fps, mapping=mapping, REACT_SMILES=REACT_SMILES, Y=Y)
        REACTION_PANDAS = pd.DataFrame({"REACT_SMILES":REACT_SMILES, "rxn_nrj":Y})
        #save the dataframe
        REACTION_PANDAS.to_csv("reaction_SN2-20.csv")

    if REGRESSION:
        data = np.load("react_SN2-20.npz", allow_pickle=True)
        fps, mapping, REACT_SMILES, Y = data["fps"], data["mapping"], data["REACT_SMILES"], data["Y"]
        fps = fps.astype(str)
        FPS_single_smiles = np.array([''.join(row) for row in fps])

        X_train, X_test,FPS_train, FPS_test, y_train, y_test = train_test_split(REACT_SMILES,FPS_single_smiles, Y, test_size=0.10, random_state=42)
        N_train = [2**i for i in range(5, 11)]
        N_train.append(len(X_train))

        KNN, KRR = False, False
        errors_KNN = []
        if KNN:
            try:
                data = np.load("learning_curve_KNN.npz", allow_pickle=True)
                errors_KNN, N_train = data["errors_KNN"], data["N_train"]
            except:
                k = 5
                for n in N_train:
                    test_preds = np.mean(regress(X_train[:n], y_train[:n], X_test, k), axis=1)
                    # compute MAE
                    mae = np.mean(np.abs(test_preds - y_test))
                    #print n and mae
                    print("n = {}, MAE = {}".format(n, mae))
                    errors_KNN.append(mae)
            
                np.savez_compressed("learning_curve_KNN.npz", errors_KNN=errors_KNN, N_train=N_train)

        if KRR:
            print("GZIP REGRESSION")
            #check if learning curve exists
            try:
                data = np.load("learning_curve_KRR.npz", allow_pickle=True)
                error_REACT_SMILES, error_FPS, N_train = data["error_REACT_SMILES"], data["error_FPS"], data["N_train"]
            except:
                error_REACT_SMILES = learning_curve(X_train, X_test, y_train, y_test)
                error_FPS= learning_curve(FPS_train, FPS_test, y_train, y_test)
                #save the learning curve
                np.savez_compressed("learning_curve_KRR.npz", error_REACT_SMILES=error_REACT_SMILES, error_FPS=error_FPS, N_train=N_train)
        
        if PLOT:
            data = np.load("learning_curve_KRR.npz", allow_pickle=True)
            error_REACT_SMILES, error_FPS, N_train = data["error_REACT_SMILES"], data["error_FPS"], data["N_train"]
            data = np.load("learning_curve_KNN.npz", allow_pickle=True)
            errors_KNN, N_train = data["errors_KNN"], data["N_train"]

            import seaborn as sns
            import matplotlib.pyplot as plt

            # Set the theme style
            sns.set_style("whitegrid")

            fig, ax = plt.subplots()
            #set xticks


            # Set both axis to log scale
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(N_train)
            ax.set_xticklabels(N_train)
            #set yticks
            ax.set_yticks([0.1, 1, 10, 100])
            ax.set_yticklabels([0.1, 1, 10, 100])
            # Set color palette
            palette = sns.color_palette("Set2")

            # Plot the data with markers and enhanced line width
            ax.plot(N_train, error_FPS, marker='o', linewidth=2, color=palette[0], label="FPS-KRR")
            ax.plot(N_train, error_REACT_SMILES, marker='s', linewidth=2, color=palette[1], label="REACT_SMILES-KRR")
            ax.plot(N_train, errors_KNN, marker='^', linewidth=2, color=palette[2], label="REACT_SMILES-KNN")


            # Set labels with enhanced font sizes
            ax.set_xlabel("$N$", fontsize=21)
            ax.set_ylabel("MAE [kcal/mol]", fontsize=21)


            # Improve legend appearance
            ax.legend(loc='center right', shadow=False, frameon=False, fontsize=12)
            #make the legend transparent
            
            #make tight layout
            plt.tight_layout()

            # Save the figure
            fig.savefig("./figures/learning_curve.png")

            


    




