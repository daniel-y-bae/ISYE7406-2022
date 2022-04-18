import os
from helpers.collect_data import DataCollector
from typing import List
from random import Random
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from helpers.lemma_token import LemmaTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score


def main():
    ## check whether to run certain functions or read from existing files
    run_vectorize = input("Vectorize data: ")
    run_corr = input("Create correlation matrix: ")
    run_pies = input("Create new pies: ")
    run_mccv = input("Run Monte Carlo cv: ")

    ## only need to run this function when creating more data for training/validation/testing
    # label_data("news_category_data.csv")

    ## create paths
    data_path = os.path.join("raw_data", "news_category_data.csv")
    output_fig_path = os.path.join("output", "figures")
    output_data_path = os.path.join("output", "data")
    
    ## get data into a dataframe
    news_df = pd.read_csv(data_path)
    ## gen_topic_titles are the titles but with the topics replaced with an...
    ## arbitrary string "abcde" so that any models don't learn to associate...
    ## specific topic names with either of the classes
    news_df["gen_topic_title"] = news_df.apply(lambda x: x["title"].lower().replace(x["topic"], "abcde"), axis=1)

    ## observe proportion of data belonging to each class for the entire sample dataset
    if run_pies == "1":
        create_pie(news_df["response"], os.path.join(output_fig_path, "classes_pie.png"), "Total")

    ## get data on the lengths of the titles
    # title_lens = news_df["title"].str.len()
    # plt.figure()
    # plt.hist(title_lens)
    # plt.title("Length of Titles")
    # plt.xlabel("Num Chars")
    # plt.ylabel("Num Titles")
    # plt.savefig(os.path.join(output_fig_path, "title_lens.png"))
    # plt.close()

    ## only need to run this function to vectorize data
    if run_vectorize == "1":
        vectorize_data(news_df, os.path.join(output_data_path, "vectorized_data.csv"))
    ## if data has already been vectorized and saved into a csv file, it can just be read from the file
    vectorized_data = pd.read_csv(os.path.join(output_data_path, "vectorized_data.csv"))
    vectorized_data["response"] = news_df["response"]

    ## plot a correlation matrix between the features and the response
    if run_corr == "1":
        plt.figure()
        plt.matshow(vectorized_data.corr(method="pearson"), fignum=1)
        plt.set_cmap("viridis")
        plt.colorbar()
        plt.title("Correlation Matrix")
        plt.savefig(os.path.join(output_fig_path, "corr_mat.png"))
        plt.close()

    ## randomly split the data into training/validation and testing sets
    x, y = vectorized_data.loc[:,vectorized_data.columns != "response"], vectorized_data["response"]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=7406)

    ## observe proportion of data belonging to each class for the training and testing datasets
    if run_pies == "1":
        create_pie(pd.Series(ytrain), os.path.join(output_fig_path, "classes_pie_train.png"), "Train")
        create_pie(pd.Series(ytest), os.path.join(output_fig_path, "classes_pie_test.png"), "Test")

    ## train a random forest model
    ## commenting out the hyperparameter tuning portion so it doesn't run everytime
    # base_rf = RandomForestClassifier(bootstrap=True, random_state=7406)
    # rf_params = {"criterion": ["gini","entropy"],
    #              "n_estimators": [150,200,250],
    #              "min_samples_leaf": [1,2,3],
    #              "max_features": ["auto","log2"],
    #              "max_samples": [0.85,0.90,0.95]}
    # cv_rf = RandomizedSearchCV(estimator=base_rf, param_distributions=rf_params, scoring="f1",
    #                            cv=3, n_iter=300, random_state=7406, verbose=0)
    # cv_rf.fit(xtrain, ytrain)
    # print(cv_rf.best_params_)
    # print(f"RF: {cv_rf.best_score_}")
    cv_rf = RandomForestClassifier(criterion="entropy", bootstrap=True, random_state=7406,
                                   n_estimators=200, min_samples_leaf=2, max_features="auto",
                                   max_samples=0.9)

    ## train a multinomial naive bayes model
    ## commenting out the hyperparameter tuning portion so it doesn't run everytime
    # base_mnb = MultinomialNB(fit_prior=True)
    # mnb_params = {"alpha": np.linspace(0.5,7.0,30,endpoint=True).tolist()}
    # cv_mnb = RandomizedSearchCV(estimator=base_mnb, param_distributions=mnb_params, scoring="f1",
    #                             cv=3, n_iter=30, random_state=7406, verbose=0)
    # cv_mnb.fit(xtrain, ytrain)
    # print(cv_mnb.best_params_)
    # print(f"MNB: {cv_mnb.best_score_}")
    cv_mnb = MultinomialNB(alpha=6.78, fit_prior=True)

    ## find the mean and standard deviations of the f1-scores of each of the models...
    ## on the training/validation data using cross-validation
    ## use Monte Carlo CV in order to make the most of a smaller data set
    if run_mccv == "1":
        model_names = ["Random Forest","Multinomial Naive Bayes","Random"]
        models = [cv_rf,cv_mnb]
        mccv_results = [[0]*(len(models)+1)]*100
        rf_feature_importances = [0]*100
        mnb_priors = [0]*100
        mnb_c0_feature_importances = [0]*100
        mnb_c1_feature_importances = [0]*100
        for i in range(100):
            curr_iter_results = [0]*(len(models)+1)
            temp_xtrain, temp_xtest, temp_ytrain, temp_ytest = train_test_split(x, y, test_size=0.3, random_state=i)
            temp_class_counts = Counter(temp_ytest)
            temp_0_pct = temp_class_counts[0] / (temp_class_counts[0] + temp_class_counts[1])
            temp_1_pct = temp_class_counts[1] / (temp_class_counts[0] + temp_class_counts[1])
            for j in range(len(models)):
                model = models[j]
                model.fit(temp_xtrain, temp_ytrain)
                model_pred = model.predict(temp_xtest)
                model_f1 = f1_score(y_true=temp_ytest, y_pred=model_pred, average="binary")
                curr_iter_results[j] = model_f1

                if isinstance(model, RandomForestClassifier):
                    rf_feature_importances[i] = model.feature_importances_
                elif isinstance(model, MultinomialNB):
                    mnb_priors[i] = [temp_0_pct,temp_1_pct]
                    mnb_c0_feature_importances[i] = model.feature_log_prob_[0,:]
                    mnb_c1_feature_importances[i] = model.feature_log_prob_[1,:]

            random_pred = random_classifier([0,1], [temp_0_pct,temp_1_pct], len(temp_ytest))
            random_f1 = f1_score(y_true=temp_ytest, y_pred=random_pred, average="binary")
            curr_iter_results[-1] = random_f1
            mccv_results[i] = curr_iter_results

        ## output Monte Carlo cross-validation results to csv
        mccv_df = pd.DataFrame(data=mccv_results, columns=model_names)
        mccv_df.to_csv(os.path.join(output_data_path, "mccv_results.csv"), header=True, index=False)

        ## create boxplots of the cross-validated f1-scores of each of the models
        plt.figure()
        plt.boxplot(mccv_df, labels=mccv_df.columns)
        plt.ylabel("F1-Score")
        plt.title("Monte Carlo Cross-Validated F1-Scores")
        plt.savefig(os.path.join(output_fig_path, "mccv_boxplot.png"))
        plt.close()

        ## output random forest feature cross-validated feature importances to csv
        rf_feature_importances_df = pd.DataFrame(data=np.row_stack(rf_feature_importances), columns=x.columns)
        rf_feature_importances_df.to_csv(os.path.join(output_data_path, "rf_feature_importances.csv"), header=True, index=False)

        ## plot the random forest feature importances based on mean decrease in impurity
        mdi = rf_feature_importances_df.mean(axis=0).sort_values(ascending=False).head(10)
        plt.figure()
        plt.bar(x=mdi.index, height=mdi)
        plt.ylabel("Mean Decrease in Impurity")
        plt.title("Monte Carlo Cross-Validated MDI")
        plt.savefig(os.path.join(output_fig_path, "mdi_bar.png"))
        plt.close()

        ## output multinomial nb cross-validated feature importances by class to csv
        mnb_priors = np.row_stack(mnb_priors)
        mnb_c0_feature_importances = np.row_stack(mnb_c0_feature_importances)
        mnb_c1_feature_importances = np.row_stack(mnb_c1_feature_importances)
        mnb_priors_df = pd.DataFrame(data=mnb_priors, columns=["class_0_priors","class_1_priors"])
        mnb_c0_feature_importances_df = pd.DataFrame(data=mnb_c0_feature_importances, columns=x.columns)
        mnb_c1_feature_importances_df = pd.DataFrame(data=mnb_c1_feature_importances, columns=x.columns)
        mnb_priors_df.to_csv(os.path.join(output_data_path, "mnb_class_priors.csv"))
        mnb_c0_feature_importances_df.to_csv(os.path.join(output_data_path, "mnb_c0_feature_importances.csv"),
                                            header=True, index=False)
        mnb_c1_feature_importances_df.to_csv(os.path.join(output_data_path, "mnb_c1_feature_importances.csv"),
                                            header=True, index=False)

        ## plot the multinomial nb feature importances based on probability for each class
        mnb_c0_prior = mnb_priors_df["class_0_priors"].mean()
        mnb_c1_prior = mnb_priors_df["class_1_priors"].mean()
        c0_feature_prob = mnb_c0_feature_importances_df.mean(axis=0).sort_values(ascending=False)
        c1_feature_prob = mnb_c1_feature_importances_df.mean(axis=0).sort_values(ascending=False)
        odds_c0 = c1_feature_prob / (c0_feature_prob*(mnb_c0_prior/mnb_c1_prior))
        odds_c1 = c0_feature_prob / (c1_feature_prob*(mnb_c0_prior/mnb_c1_prior))
        odds_c0 = odds_c0.sort_values(ascending=False).head(10)
        odds_c1 = odds_c1.sort_values(ascending=False).head(10)

        plt.figure()
        plt.bar(x=odds_c0.index, height=odds_c0)
        plt.ylabel("Class 0 Odds")
        plt.title("Monte Carlo Cross-Validated Odds of Class 0")
        plt.savefig(os.path.join(output_fig_path, "c0_feature_odds.png"))
        plt.close()
        
        plt.figure()
        plt.bar(x=odds_c1.index, height=odds_c1)
        plt.ylabel("Class 1 Odds")
        plt.title("Monte Carlo Cross-Validated Odds of Class 1")
        plt.savefig(os.path.join(output_fig_path, "c1_feature_odds.png"))
        plt.close()
    
    ## final evaluation of the models on the reserved testing data set
    cv_rf.fit(xtrain, ytrain)
    rf_test_pred = cv_rf.predict(xtest)
    rf_test_f1 = f1_score(y_true=ytest, y_pred=rf_test_pred, average="binary")
    rf_test_results = news_df.iloc[xtest.index,:].copy()
    rf_test_results["prediction"] = rf_test_pred
    rf_test_results.to_csv(os.path.join(output_data_path, "rf_test_results.csv"), header=True, index=False)
    print(rf_test_f1)

    cv_mnb.fit(xtrain, ytrain)
    mnb_test_pred = cv_mnb.predict(xtest)
    mnb_test_f1 = f1_score(y_true=ytest, y_pred=mnb_test_pred, average="binary")
    mnb_test_results = news_df.iloc[xtest.index,:].copy()
    mnb_test_results["prediction"] = mnb_test_pred
    mnb_test_results.to_csv(os.path.join(output_data_path, "mnb_test_results.csv"), header=True, index=False)
    print(mnb_test_f1)

    test_class_counts = Counter(ytest)
    test_0_pct = test_class_counts[0] / (test_class_counts[0] + test_class_counts[1])
    test_1_pct = test_class_counts[1] / (test_class_counts[0] + test_class_counts[1])
    random_test_pred = random_classifier([0,1], [test_0_pct,test_1_pct], len(ytest))
    random_test_f1 = f1_score(y_true=ytest, y_pred=random_test_pred, average="binary")
    random_test_results = news_df.iloc[xtest.index,:].copy()
    random_test_results["prediction"] = random_test_pred
    random_test_results.to_csv(os.path.join(output_data_path, "random_test_results.csv"), header=True, index=False)
    print(random_test_f1)


def label_data(output_file: str) -> None:
    """
    Calls the data_collector class from the collect_data.py module.

    Parameter(s)
    ------------
    output_file: str
        The name of the csv file to pass to the data_collector which is where the labelled data will be saved.
        
    Returns
    -------
    None
    """
    new_data_collector = DataCollector(output_file)
    new_data_collector.check_old_titles()
    new_data_collector.get_user_input()
    new_data_collector.output_to_file()


def create_pie(data_series: pd.Series, save_path: os.path, dataset: str) -> None:
    """
    Creates and saves a basic pie chart of the proportion of the two classes in the data passed.

    Parameter(s)
    ------------
    data_series: pd.Series
        Pandas Series containing the class labels.

    save_path: os.path
        Path to save output to.

    dataset: str
        String that describes the dataset (to be used to name the figure). For example "Train".

    Returns
    -------
    None
    """

    class_counts = data_series.groupby(data_series).count()
    plt.figure()
    plt.pie(class_counts, labels=class_counts.index, autopct="%1.1f%%")
    plt.title(f"Class Share - {dataset}")
    plt.savefig(save_path)
    plt.close()


def vectorize_data(data_df: pd.DataFrame, save_path: os.path) -> None:
    """
    Vectorize gen_topic_title. 

    Remove words/features that appear in more than 97% of examples or fewer than 3% of examples.
    Not removing any stop words, instead will learn what the stop words are based on their frequency in the training data.
    For example, common stop words such as "which" might actually be important for identifying
    clickbait, hence we would not want to remove them unless they appear extremely frequently or infrequently.

    Parameter(s)
    ------------
    data_df: pd.DataFrame
        Pandas DataFrame containing all of the labelled data.

    save_path: os.path
        Path to save output to.
        
    Returns
    -------
    None
    """
    vectorizer = CountVectorizer(lowercase=True, tokenizer=LemmaTokenizer(),
                                 stop_words=None, analyzer="word", 
                                 max_df=0.97, min_df=0.03)
    vect_matrix = vectorizer.fit_transform(data_df["gen_topic_title"])
    vect_array = vect_matrix.toarray()
    vect_df = pd.DataFrame(data=vect_array, columns=vectorizer.get_feature_names_out())
    vect_df.to_csv(save_path, header=True, index=False)


def random_classifier(p: List[int], w: List[float], n: int) -> List[int]:
    """
    Randomly chooses n samples from (p)opulation with replacement. 
    The probability of a choice in (p)opulation being chosen is influenced by (w)eights.

    Parameter(s)
    ------------
    p: List[int]
        A list of possible choices to sample from.

    w: List[float]
        The probability of a choice from (p)opulation being picked.
        The values in this list should sum to 1.0.

    n: int
        The number of samples to select from (p)opulation with replacement.
        Essentially, n is the length of the list being returned.
        
    Returns
    -------
    random_classifications: List[int]
        A list of length n with random selections from (p)opulation with replacement and influenced by (w)eights.    
    """

    assert sum(w) == 1.0, "(w)eights must sum to 1.0"

    temp_rng = Random(7406)
    random_classifications = temp_rng.choices(population=p, weights=w, k=n)
    return random_classifications


if __name__ == "__main__":
    main()