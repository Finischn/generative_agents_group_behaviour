import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Add SpacyTextBlob to the spaCy pipeline
nlp.add_pipe('spacytextblob')

# Function to perform sentiment analysis on each utterance in the "Details" column using SpacyTextBlob
def analyze_sentiments(details):
    """
    Analyze sentiments for each utterance in a conversation.

    This function processes a string of conversation details, splitting it into individual
    utterances, identifying the speaker, and computing sentiment scores for each utterance
    using SpacyTextBlob.

    Parameters:
    details (str): A semicolon-separated string of conversation details, where each utterance 
                   is in the format "Speaker: Message".

    Returns:
    list: A list of tuples, where each tuple contains:
          - The utterance (str)
          - A dictionary with keys:
              - 'speaker': The name of the speaker (str)
              - 'polarity': The sentiment polarity of the utterance (float, range [-1.0, 1.0])
              - 'subjectivity': The sentiment subjectivity of the utterance (float, range [0.0, 1.0])
    """
    # Split the details by the semicolon to separate each utterance
    utterances = details.split(";")
    sentiments = []
    for utterance in utterances:
        if utterance.strip():  # Ensure the utterance is not just whitespace
            # Split the utterance into speaker and message
            speaker, message = utterance.split(":", 1)
            speaker = speaker.strip()  # Remove any leading/trailing whitespace from the speaker's name
            message = message.strip()  # Remove any leading/trailing whitespace from the message

            # Process the message with spaCy
            doc = nlp(message)
            # Compute sentiment score using SpacyTextBlob
            score = {
                'speaker': speaker,
                'polarity': doc._.blob.polarity,        # Access sentiment polarity
                'subjectivity': doc._.blob.subjectivity  # Access sentiment subjectivity
            }
            sentiments.append((message, score))

    return sentiments


def average_polarity(sentiments):
    """
    Compute the average polarity from a list of sentiment scores.

    Parameters:
    sentiments (list): A list of tuples where each tuple contains:
                       - The utterance (str)
                       - A dictionary with keys:
                         - 'polarity': Sentiment polarity (float, range [-1.0, 1.0])

    Returns:
    float or None: The average polarity, or None if no polarities are provided.
    """
    polarity_scores = [sentiment[1]['polarity'] for sentiment in sentiments]
    if polarity_scores:
        return sum(polarity_scores) / len(polarity_scores)
    else:
        return None



def anova_sentiment(df_names, combined_df):
    """
    Perform ANOVA on the Average Polarity of the specified DataFrames.

    Parameters:
    df_names (list): List of DataFrame names to include in the analysis.
    combined_df (pd.DataFrame): DataFrame containing 'Source' and 'AveragePolarity' columns.

    Returns:
    pd.DataFrame: ANOVA table
    """
    # Filter the combined DataFrame for the specified DataFrame names
    filtered_df = combined_df[combined_df['Source'].isin(df_names)]

    if filtered_df.empty:
        raise ValueError("No matching DataFrames found for the provided names in combined_df.")

    # Step 2: Perform ANOVA
    # The model formula specifies 'AveragePolarity' as the dependent variable
    # and 'Source' as the independent variable (factor)
    model = ols('AveragePolarity ~ Source', data=filtered_df).fit()

    # Generate the ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)

    return anova_table



def compare_groups(combined_df, reference_groups, source_groups):
    """
    Compare the Average Polarity between reference groups and source groups using t-tests.

    Parameters:
    combined_df (pd.DataFrame): DataFrame containing 'Source' and 'AveragePolarity' columns.
    reference_groups (list): List of reference group names to compare against.
    source_groups (list): List of source group names to compare with reference groups.

    Returns:
    pd.DataFrame: Results of the comparisons with t-statistics and p-values.
    """
    results = []

    for ref_group in reference_groups:
        for source_group in source_groups:
            ref_data = combined_df[combined_df['Source'] == ref_group]['AveragePolarity']
            source_data = combined_df[combined_df['Source'] == source_group]['AveragePolarity']

            # Perform the t-test
            t_stat, p_value = stats.ttest_ind(ref_data, source_data, equal_var=False)

            # Append the results
            results.append({
                'Reference Group': ref_group,
                'Source Group': source_group,
                'T-Statistic': t_stat,
                'P-Value': p_value
            })

    # Return results as a DataFrame
    return pd.DataFrame(results)



def plot_boxplots_per_reference_group(combined_df, reference_groups, source_groups):
    """
    Generate one boxplot per reference group showing the reference group and its source groups.
    
    Parameters:
    combined_df (pd.DataFrame): DataFrame containing 'Source' and 'AveragePolarity'.
    reference_groups (list): List of reference group names.
    source_groups (list): List of source group names.
    """
    for ref_group in reference_groups:
        # Prepare data for the reference group and all source groups
        plot_data = []
        group_labels = []

        # Add the reference group data
        ref_data = combined_df[combined_df['Source'] == ref_group]['AveragePolarity']
        plot_data.extend(ref_data)
        group_labels.extend([ref_group] * len(ref_data))

        # Add data for each source group
        for source_group in source_groups:
            source_data = combined_df[combined_df['Source'] == source_group]['AveragePolarity']
            plot_data.extend(source_data)
            group_labels.extend([source_group] * len(source_data))

        # Create a DataFrame for plotting
        data_to_plot = pd.DataFrame({'Polarity': plot_data, 'Group': group_labels})

        # Plot the boxplot
        plt.figure(figsize=(20, 8))
        sns.boxplot(data=data_to_plot, x='Group', y='Polarity', hue= 'Group', palette="Set2")
        plt.title(f"Polarity Comparison for Reference Group: {ref_group}")
        plt.xlabel("Group")
        plt.ylabel("Average Polarity")
        plt.xticks(rotation=30)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()